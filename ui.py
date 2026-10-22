"""Small, local presentation helpers. No external fonts or assets."""

from html import escape
from pathlib import Path
import base64

import streamlit as st


def setup_ui(accent="#126b62", soft="#e6f3ef"):
    css = Path(__file__).with_name("styles.css").read_text()
    font = base64.b64encode((Path(__file__).parent / 'fonts' / 'stix-two-text.ttf').read_bytes()).decode()
    face = "@font-face{font-family:'STIX Two Text';src:url(data:font/ttf;base64," + font + ") format('truetype');font-weight:400 700;font-display:swap;}"
    st.markdown(f"<style>{face}:root {{ --accent: {accent}; --accent-soft: {soft}; }}\n{css}</style>", unsafe_allow_html=True)


def hero(name, eyebrow, description, tags):
    pills = "".join(f'<span class="hero-pill">{escape(tag)}</span>' for tag in tags)
    st.markdown(
        f'<header class="hero"><div class="eyebrow">{escape(eyebrow)}</div>'
        f'<h1>{escape(name)}<span class="brand-dot">.</span></h1>'
        f'<p class="hero-description">{escape(description)}</p>'
        f'<div class="hero-pills">{pills}</div><div class="hero-orbit" aria-hidden="true"></div></header>',
        unsafe_allow_html=True,
    )


def section(number, title, description=""):
    st.markdown(
        f'<div class="section-heading"><span class="section-number">{escape(number)}</span>'
        f'<h2>{escape(title)}</h2></div>'
        + (f'<p class="section-description">{escape(description)}</p>' if description else ""),
        unsafe_allow_html=True,
    )


def empty_state(title, description):
    st.markdown(f'<div class="empty-state"><div class="eyebrow">READY WHEN YOU ARE</div>'
                f'<h3>{escape(title)}</h3><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def badges(items):
    st.markdown('<div class="evidence-badges">' + ''.join(
        f'<span>{escape(str(item))}</span>' for item in items
    ) + '</div>', unsafe_allow_html=True)
