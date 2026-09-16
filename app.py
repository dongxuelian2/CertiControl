"""CertiControl Streamlit application.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import streamlit as st
import sympy as sp

from app_helpers import eigenvalue_figure, open_closed_loop_figure
from certicontrol import (
    AnalysisOptions,
    LTISystem,
    TolerancePolicy,
    analyze_system,
    curated_examples,
    parse_matrix,
)
from certicontrol.examples import ExampleSystem, example_matrix_text, matrix_to_input_text
from certicontrol.reporting import format_scalar, format_warning, matrix_to_latex, to_json, to_markdown


def _prefer_exact(exact: Any, numeric: Any) -> Any:
    return exact if exact is not None else numeric


def _init_state() -> None:
    if st.session_state.get("_certicontrol_initialized"):
        return
    example = curated_examples()["Stable minimal second-order"]
    texts = example_matrix_text(example)
    for name in ("A", "B", "C", "D"):
        st.session_state[f"{name}_text"] = texts[name]
    st.session_state["Q_text"] = matrix_to_input_text(example.default_Q)
    st.session_state["R_text"] = matrix_to_input_text(example.default_R)
    st.session_state["tolerance_mode"] = "Automatic"
    st.session_state["custom_tolerance"] = 1e-9
    st.session_state["analysis"] = None
    st.session_state["input_error"] = None
    st.session_state["lqr_input_error"] = None
    st.session_state["_certicontrol_initialized"] = True


def _load_example(example: ExampleSystem) -> None:
    texts = example_matrix_text(example)
    for name in ("A", "B", "C", "D"):
        st.session_state[f"{name}_text"] = texts[name]
    st.session_state["Q_text"] = matrix_to_input_text(example.default_Q)
    st.session_state["R_text"] = matrix_to_input_text(example.default_R)
    if example.recommended_absolute_tolerance is None:
        st.session_state["tolerance_mode"] = "Automatic"
    else:
        st.session_state["tolerance_mode"] = "Custom"
        st.session_state["custom_tolerance"] = float(example.recommended_absolute_tolerance)
    st.session_state["analysis"] = None
    st.session_state["input_error"] = None
    st.session_state["lqr_input_error"] = None


def _parse_optional(text: str) -> Any | None:
    return None if not text.strip() else parse_matrix(text)


def _status_box(status: str, verdict: str | None = None) -> None:
    text = status if verdict is None else f"{status} — {verdict}"
    if status == "PASS":
        st.success(text)
    elif status in {"FAIL", "ERROR"}:
        st.error(text)
    elif status == "INCONCLUSIVE":
        st.warning(text)
    else:
        st.info(text)


def _render_record_header(analysis: Any, key: str) -> Any | None:
    record = analysis.records[key]
    verdict = None if record.certificate is None else record.certificate.verdict
    _status_box(record.status, verdict)
    if record.reason:
        st.caption(record.reason)
    if record.error:
        st.error(record.error)
    return record.certificate


def _show_matrix(label: str, matrix: Any | None) -> None:
    if matrix is None:
        return
    st.markdown(f"**{label}**")
    st.latex(matrix_to_latex(matrix))


def _warning_center(analysis: Any) -> None:
    st.subheader("Warnings")
    if not analysis.warnings and not analysis.warning_codes:
        st.caption("No warnings reported.")
        return
    shown: set[str] = set()
    for item in analysis.warning_codes:
        message = format_warning(item["code"], item.get("source"))
        if message not in shown:
            st.warning(message)
            shown.add(message)
    for warning in analysis.warnings:
        if warning not in shown:
            st.warning(warning)
            shown.add(warning)


def _eigen_rows(values: Any) -> list[dict[str, float]]:
    array = np.asarray(values).reshape(-1)
    return [
        {"real": float(np.real(value)), "imaginary": float(np.imag(value))}
        for value in array
    ]


def _render_overview(analysis: Any) -> None:
    st.header("Overview")
    summary = analysis.summary()
    a, b, c = st.columns(3)
    a.metric("States", summary["states"])
    b.metric("Inputs", summary["inputs"])
    c.metric("Outputs", summary["outputs"] if summary["outputs"] is not None else "—")

    st.subheader("Certificate status")
    cols = st.columns(5)
    for col, key, label in zip(
        cols,
        ("controllability", "observability", "stability", "kalman", "lqr"),
        ("Controllability", "Observability", "Hurwitz", "Kalman", "LQR"),
    ):
        col.metric(label, analysis.status(key))

    kalman = analysis.certificates.get("kalman")
    if kalman is not None:
        d = kalman.diagnostics
        st.subheader("Structural dimensions")
        cols = st.columns(4)
        cols[0].metric("Controllable / Observable", d.get("controllable_observable_dimension"))
        cols[1].metric("Controllable / Unobservable", d.get("controllable_unobservable_dimension"))
        cols[2].metric("Uncontrollable / Observable", d.get("uncontrollable_observable_dimension"))
        cols[3].metric("Uncontrollable / Unobservable", d.get("uncontrollable_unobservable_dimension"))

    _warning_center(analysis)
    exact_parts = [
        name
        for name in ("A", "B", "C", "D")
        if getattr(analysis.system, name) is not None and getattr(analysis.system, f"exact_{name}") is not None
    ]
    if exact_parts:
        st.caption("Exact rational input detected: " + ", ".join(exact_parts))


def _render_pbh_failure(pbh: Any, *, output: bool) -> None:
    if pbh is None or pbh.passed is not False:
        return
    failures = pbh.evidence.get("exact_failures") or pbh.evidence.get("numerical_failures") or []
    if not failures:
        return
    failure = failures[0]
    st.subheader("Failure witness")
    st.write("Offending mode:", format_scalar(failure.get("eigenvalue")))
    _show_matrix("Witness", failure.get("witness"))
    st.write("Eigenvector residual:", format_scalar(failure.get("eigenvector_residual")))
    residual_key = "output_null_residual" if output else "input_orthogonality_residual"
    st.write("Output/input residual:", format_scalar(failure.get(residual_key)))


def _render_controllability(analysis: Any) -> None:
    st.header("Controllability")
    cert = _render_record_header(analysis, "controllability")
    if cert is None:
        return
    d = cert.diagnostics
    rank = d.get("exact_rank") if d.get("exact_rank") is not None else d.get("numerical_rank")
    x, y, z = st.columns(3)
    x.metric("Rank", f"{rank} / {d.get('n')}")
    y.metric("PBH", "PASS" if cert.evidence["pbh"].passed else "FAIL")
    z.metric("Evidence", "EXACT + numerical" if d.get("exact_rank") is not None else "NUMERICAL")
    st.write("Singular values:", np.asarray(d.get("singular_values")))
    st.write("Active tolerance:", d.get("tolerance"))
    with st.expander("Mathematical evidence"):
        _show_matrix(
            "Controllability matrix",
            _prefer_exact(cert.evidence.get("controllability_matrix_exact"), cert.evidence.get("controllability_matrix")),
        )
        st.json(cert.evidence["pbh"].to_dict(json_safe=True), expanded=False)
    _render_pbh_failure(cert.evidence.get("pbh"), output=False)


def _render_observability(analysis: Any) -> None:
    st.header("Observability")
    cert = _render_record_header(analysis, "observability")
    if cert is None:
        return
    d = cert.diagnostics
    rank = d.get("exact_rank") if d.get("exact_rank") is not None else d.get("numerical_rank")
    x, y, z = st.columns(3)
    x.metric("Rank", f"{rank} / {d.get('n')}")
    y.metric("PBH", "PASS" if cert.evidence["pbh"].passed else "FAIL")
    z.metric("Evidence", "EXACT + numerical" if d.get("exact_rank") is not None else "NUMERICAL")
    st.write("Singular values:", np.asarray(d.get("singular_values")))
    st.write("Active tolerance:", d.get("tolerance"))
    with st.expander("Mathematical evidence"):
        _show_matrix(
            "Observability matrix",
            _prefer_exact(cert.evidence.get("observability_matrix_exact"), cert.evidence.get("observability_matrix")),
        )
        st.json(cert.evidence["pbh"].to_dict(json_safe=True), expanded=False)
    _render_pbh_failure(cert.evidence.get("pbh"), output=True)


def _render_stability(analysis: Any) -> None:
    st.header("Stability / Lyapunov")
    cert = _render_record_header(analysis, "stability")
    if cert is None:
        return
    spectral = cert.evidence.get("spectral")
    lyapunov = cert.evidence.get("lyapunov")
    if spectral is not None:
        d = spectral.diagnostics
        x, y, z = st.columns(3)
        x.metric("Spectral abscissa", format_scalar(d.get("spectral_abscissa")))
        y.metric("Spectral margin", format_scalar(d.get("spectral_margin")))
        z.metric("Hurwitz", spectral.verdict)
        eigs = d.get("eigenvalues", [])
        st.plotly_chart(eigenvalue_figure(eigs, title="Open-loop poles"), use_container_width=True)
        st.dataframe(_eigen_rows(eigs), use_container_width=True)
    if lyapunov is not None:
        st.subheader("Lyapunov certificate")
        d = lyapunov.diagnostics
        x, y, z = st.columns(3)
        x.metric("lambda_min(P)", format_scalar(d.get("lambda_min_P")))
        y.metric("Relative residual", format_scalar(d.get("relative_residual")))
        z.metric("Exact certificate", format_scalar(d.get("exact_certificate_passed")))
        P = _prefer_exact(lyapunov.evidence.get("P_exact"), lyapunov.evidence.get("P"))
        with st.expander("Mathematical evidence"):
            _show_matrix("P", P)
            st.write("Symmetry residual:", d.get("symmetry_residual"))
            st.write("Exact leading principal minors:", d.get("P_leading_principal_minors"))


def _render_lqr(analysis: Any) -> None:
    st.subheader("LQR certificate")
    cert = _render_record_header(analysis, "lqr")
    if cert is None:
        if analysis.records["lqr"].status == "NOT_RUN":
            st.caption("Configure Q and R above and run analysis to synthesize state feedback.")
        return
    d = cert.diagnostics
    top = st.columns(4)
    top[0].metric("Q PSD", format_scalar(d.get("Q_psd_status")))
    top[1].metric("R PD", format_scalar(d.get("R_pd_status")))
    top[2].metric("Stabilizable", format_scalar(d.get("stabilizable")))
    top[3].metric("Detectable", format_scalar(d.get("detectable")))
    mid = st.columns(3)
    mid[0].metric("CARE residual", format_scalar(d.get("care_relative_residual")))
    mid[1].metric("Closed-loop identity", format_scalar(d.get("closed_loop_identity_relative_residual")))
    mid[2].metric("Closed-loop abscissa", format_scalar(d.get("closed_loop_spectral_abscissa")))

    stability = analysis.certificates.get("stability")
    if stability is not None:
        spectral = stability.evidence.get("spectral")
        open_eigs = [] if spectral is None else spectral.diagnostics.get("eigenvalues", [])
        closed_eigs = d.get("closed_loop_eigenvalues", [])
        st.plotly_chart(open_closed_loop_figure(open_eigs, closed_eigs), use_container_width=True)

    with st.expander("Mathematical evidence"):
        _show_matrix("P", cert.evidence.get("P"))
        _show_matrix("K", cert.evidence.get("K"))
        _show_matrix("A_closed_loop", cert.evidence.get("A_closed_loop"))
        st.write("P symmetry residual:", d.get("P_symmetry_residual"))
        st.write("Gain consistency residual:", d.get("gain_relative_residual"))


def _render_kalman(analysis: Any) -> None:
    st.header("Kalman Structure")
    cert = _render_record_header(analysis, "kalman")
    if cert is None:
        return
    d = cert.diagnostics
    a, b, c = st.columns(3)
    a.metric("Reachable dimension", d.get("reachable_dimension"))
    b.metric("Unobservable dimension", d.get("unobservable_dimension"))
    c.metric("Intersection dimension", d.get("intersection_dimension"))

    st.subheader("Four-part structure")
    row1 = st.columns(2)
    row1[0].metric("Controllable / Observable", d.get("controllable_observable_dimension"))
    row1[1].metric("Controllable / Unobservable", d.get("controllable_unobservable_dimension"))
    row2 = st.columns(2)
    row2[0].metric("Uncontrollable / Observable", d.get("uncontrollable_observable_dimension"))
    row2[1].metric("Uncontrollable / Unobservable", d.get("uncontrollable_unobservable_dimension"))

    st.write("State transformation:  x = T z")
    st.write("cond(T):", d.get("transformation_condition_number"))
    residuals = {
        "A structural zero": d.get("A_structural_zero_relative_residual"),
        "B reachable support": d.get("B_structural_zero_relative_residual"),
        "C unobservable annihilation": d.get("C_structural_zero_relative_residual"),
        "Similarity A": d.get("similarity_relative_residual_A"),
        "Similarity B": d.get("similarity_relative_residual_B"),
        "Similarity C": d.get("similarity_relative_residual_C"),
    }
    st.dataframe([{"check": key, "relative residual": value} for key, value in residuals.items()], use_container_width=True)

    with st.expander("Transformation and transformed matrices"):
        _show_matrix("T", cert.evidence.get("T"))
        _show_matrix("A_bar", cert.evidence.get("A_transformed"))
        _show_matrix("B_bar", cert.evidence.get("B_transformed"))
        _show_matrix("C_bar", cert.evidence.get("C_transformed"))
        if cert.evidence.get("D_transformed") is not None:
            _show_matrix("D_bar = D", cert.evidence.get("D_transformed"))
        st.caption("Complement bases and T are generally non-unique; the canonical information is R, N, R ∩ N, and the structural dimensions.")


def _render_export(analysis: Any) -> None:
    st.header("Export")
    markdown = to_markdown(analysis)
    json_text = to_json(analysis, indent=2)
    left, right = st.columns(2)
    left.download_button(
        "Download Markdown Report",
        data=markdown,
        file_name="certicontrol_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
    right.download_button(
        "Download JSON Certificate",
        data=json_text,
        file_name="certicontrol_certificate.json",
        mime="application/json",
        use_container_width=True,
    )
    with st.expander("Preview Markdown"):
        st.code(markdown, language="markdown")
    with st.expander("Preview JSON"):
        st.json(json.loads(json_text), expanded=False)


def main() -> None:
    st.set_page_config(page_title="CertiControl", page_icon="✓", layout="wide")
    _init_state()

    st.title("CertiControl")
    st.caption("Inspect the evidence behind linear control analysis.")

    examples = curated_examples()
    with st.sidebar:
        st.header("System")
        selected_name = st.selectbox("Load example", list(examples), key="example_selector")
        st.caption(examples[selected_name].description)
        if st.button("Load selected example", use_container_width=True):
            _load_example(examples[selected_name])
        if st.button("Reset to stable example", use_container_width=True):
            _load_example(examples["Stable minimal second-order"])

        st.text_area("A  (n × n)", key="A_text", height=110)
        st.text_area("B  (n × m)", key="B_text", height=90)
        st.text_area("C  (p × n, optional)", key="C_text", height=90)
        st.text_area("D  (p × m, optional)", key="D_text", height=75)

        st.header("Analysis")
        run_controllability = st.checkbox("Controllability", value=True)
        run_observability = st.checkbox("Observability", value=True)
        run_stability = st.checkbox("Stability / Lyapunov", value=True)
        run_kalman = st.checkbox("Kalman structure", value=True)
        run_lqr = st.checkbox("LQR / CARE", value=True)

        with st.expander("Advanced settings"):
            tolerance_mode = st.radio(
                "Numerical tolerance mode",
                ["Automatic", "Custom"],
                key="tolerance_mode",
            )
            if tolerance_mode == "Custom":
                st.number_input(
                    "Absolute singular-value tolerance",
                    min_value=0.0,
                    key="custom_tolerance",
                    format="%.3e",
                )
            st.caption("Residual checks remain scale-aware inside the mathematical certificate modules.")

        analyze_clicked = st.button("Analyze system", type="primary", use_container_width=True)

    tabs = st.tabs(
        ["Overview", "Controllability", "Observability", "Stability", "LQR", "Kalman Structure", "Export"]
    )

    with tabs[4]:
        st.header("LQR / CARE")
        st.caption("Continuous-time infinite-horizon state feedback only. Configure Q and R here, then run analysis.")
        if st.button("Use identity Q and R", key="identity_weights"):
            try:
                A_guess = parse_matrix(st.session_state["A_text"])
                B_guess = parse_matrix(st.session_state["B_text"])
                st.session_state["Q_text"] = matrix_to_input_text(sp.eye(A_guess.rows))
                st.session_state["R_text"] = matrix_to_input_text(sp.eye(B_guess.cols))
            except ValueError as exc:
                st.warning(f"Cannot infer identity weights: {exc}")
        st.text_area("Q  (n × n, PSD)", key="Q_text", height=90)
        st.text_area("R  (m × m, PD)", key="R_text", height=75)

    if analyze_clicked:
        st.session_state["input_error"] = None
        st.session_state["lqr_input_error"] = None
        try:
            A = parse_matrix(st.session_state["A_text"])
            B = parse_matrix(st.session_state["B_text"])
            C = _parse_optional(st.session_state["C_text"])
            D = _parse_optional(st.session_state["D_text"])
            system = LTISystem(A, B, C=C, D=D)
        except ValueError as exc:
            st.session_state["analysis"] = None
            st.session_state["input_error"] = str(exc)
        else:
            policy = (
                TolerancePolicy(absolute=float(st.session_state["custom_tolerance"]))
                if st.session_state["tolerance_mode"] == "Custom"
                else TolerancePolicy()
            )
            Q = R = None
            lqr_weights_valid = True
            if run_lqr:
                try:
                    Q = _parse_optional(st.session_state["Q_text"])
                    R = _parse_optional(st.session_state["R_text"])
                except ValueError as exc:
                    lqr_weights_valid = False
                    st.session_state["lqr_input_error"] = str(exc)
            options = AnalysisOptions(
                run_controllability=run_controllability,
                run_observability=run_observability,
                run_stability=run_stability,
                run_kalman=run_kalman,
                run_lqr=run_lqr and lqr_weights_valid,
            )
            with st.spinner("Analyzing system..."):
                st.session_state["analysis"] = analyze_system(
                    system,
                    options=options,
                    Q=Q,
                    R=R,
                    tolerance_policy=policy,
                )

    if st.session_state.get("input_error"):
        st.error(f"Invalid system input: {st.session_state['input_error']}")
    if st.session_state.get("lqr_input_error"):
        with tabs[4]:
            st.error(f"Invalid LQR weighting matrix: {st.session_state['lqr_input_error']}")

    analysis = st.session_state.get("analysis")
    if analysis is None:
        with tabs[0]:
            st.info("Load an example or enter A/B/C/D, then click **Analyze system**.")
        for tab in (tabs[1], tabs[2], tabs[3], tabs[5], tabs[6]):
            with tab:
                st.caption("Run analysis to populate this panel.")
        with tabs[4]:
            st.caption("Run analysis to populate the LQR certificate panel.")
    else:
        with tabs[0]:
            _render_overview(analysis)
        with tabs[1]:
            _render_controllability(analysis)
        with tabs[2]:
            _render_observability(analysis)
        with tabs[3]:
            _render_stability(analysis)
        with tabs[4]:
            st.divider()
            _render_lqr(analysis)
        with tabs[5]:
            _render_kalman(analysis)
        with tabs[6]:
            _render_export(analysis)

    st.divider()
    st.caption(
        "CertiControl certificates are inspectable computational evidence. "
        "They are not formal proofs or safety-critical engineering certification."
    )


if __name__ == "__main__":
    main()