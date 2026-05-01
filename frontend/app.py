"""
frontend/app.py

GlamScan — Streamlit web interface.

Run:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st
from PIL import Image

API_BASE = "http://localhost:8000"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GlamScan 💄",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'Playfair Display', serif; }

  .glam-header {
    background: linear-gradient(135deg, #1a0a0f 0%, #3d1020 50%, #1a0a0f 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 2rem;
  }
  .glam-header h1 { color: #f5c6d0; font-size: 3rem; margin: 0; }
  .glam-header p  { color: #c9848f; font-size: 1.1rem; margin-top: .5rem; }

  .product-card {
    background: #fff;
    border: 1px solid #f0e6e8;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    transition: box-shadow .2s;
  }
  .product-card:hover { box-shadow: 0 4px 20px rgba(180,80,100,.15); }

  .source-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: .75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: .05em;
  }
  .badge-sephora { background: #fce4ec; color: #c2185b; }
  .badge-amazon  { background: #fff3e0; color: #e65100; }

  .price-tag {
    font-size: 1.4rem;
    font-weight: 700;
    color: #b5293a;
  }
  .similarity-bar {
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, #e91e63, #f48fb1);
    margin-top: 4px;
  }
  a.buy-link {
    display: inline-block;
    background: #b5293a;
    color: white !important;
    padding: .45rem 1.2rem;
    border-radius: 8px;
    font-size: .85rem;
    font-weight: 500;
    text-decoration: none;
    margin-top: .5rem;
  }
  a.buy-link:hover { background: #8d1a28; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="glam-header">
  <h1>💄 GlamScan</h1>
  <p>Upload a cosmetic product image — find it instantly across Sephora &amp; Amazon</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Search Options")
    search_mode = st.radio(
        "Search mode",
        ["Image only", "Text only", "Image + Text"],
        index=0,
    )
    top_k = st.slider("Results to show", 3, 20, 8)
    if search_mode == "Image + Text":
        text_weight = st.slider("Text influence", 0.0, 1.0, 0.5, 0.05)
    else:
        text_weight = 0.5

    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("""
1. Upload a product photo
2. CLIP converts it to a 512-d embedding
3. FAISS finds the closest products
4. Prices from Sephora & Amazon are compared
""")

# ── Upload / Text area ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

uploaded = None
text_query = ""

with col_left:
    if search_mode in ("Image only", "Image + Text"):
        uploaded = st.file_uploader(
            "Upload product image",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded:
            st.image(uploaded, caption="Query image", use_container_width=True)

    if search_mode in ("Text only", "Image + Text"):
        text_query = st.text_input(
            "Describe the product",
            placeholder="e.g. matte red lipstick, dewy foundation",
        )

    search_btn = st.button("🔍 Search GlamScan", use_container_width=True, type="primary")

# ── Run search ─────────────────────────────────────────────────────────────────
with col_right:
    if search_btn:
        results = None
        
        # Show progress bar and status
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            import time
            start_time = time.time()
            
            # Upload phase
            status_placeholder.info("📤 Uploading image...")
            progress_bar = progress_placeholder.progress(0)
            
            if search_mode == "Image only" and uploaded:
                progress_bar.progress(10)
                status_placeholder.info("⏳ Waiting for API... (CLIP model loading on first request)")
                time.sleep(0.5)  # visual feedback
                progress_bar.progress(30)
                
                resp = requests.post(
                    f"{API_BASE}/search/image",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    params={"k": top_k},
                    timeout=120,
                )
                progress_bar.progress(70)
                
            elif search_mode == "Text only" and text_query:
                progress_bar.progress(10)
                status_placeholder.info("⏳ Waiting for API... (CLIP model loading on first request)")
                time.sleep(0.5)
                progress_bar.progress(30)
                
                resp = requests.post(
                    f"{API_BASE}/search/text",
                    json={"query": text_query, "k": top_k},
                    timeout=120,
                )
                progress_bar.progress(70)
                
            elif search_mode == "Image + Text" and uploaded and text_query:
                progress_bar.progress(10)
                status_placeholder.info("⏳ Waiting for API... (CLIP model loading on first request)")
                time.sleep(0.5)
                progress_bar.progress(30)
                
                resp = requests.post(
                    f"{API_BASE}/search/combined",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    params={"query": text_query, "k": top_k, "text_weight": text_weight},
                    timeout=120,
                )
                progress_bar.progress(70)
                
            else:
                st.warning("Please provide the required input for the selected search mode.")
                resp = None

            if resp is not None:
                progress_bar.progress(95)
                if resp.status_code == 200:
                    results = resp.json()
                    progress_bar.progress(100)
                    elapsed = time.time() - start_time
                    status_placeholder.success(f"✅ Done! Found {len(results)} products in {elapsed:.1f}s")
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
                    progress_placeholder.empty()
                    status_placeholder.empty()

        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out (120s). CLIP model may still be loading. Try again in a moment.")
            progress_placeholder.empty()
            status_placeholder.empty()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to GlamScan API. Start it with: `bash run.sh api`")
            progress_placeholder.empty()
            status_placeholder.empty()
        
        # Clear progress after showing results
        if results:
            progress_placeholder.empty()
            status_placeholder.empty()

        # ── Display results ────────────────────────────────────────────────────
        if results:
            st.markdown(f"### Found **{len(results)}** similar products")

            # Price comparison summary
            prices = [r["price_usd"] for r in results if r.get("price_usd")]
            if prices:
                min_p, max_p = min(prices), max(prices)
                c1, c2, c3 = st.columns(3)
                c1.metric("Lowest price", f"${min_p:.2f}")
                c2.metric("Highest price", f"${max_p:.2f}")
                c3.metric("Avg price", f"${sum(prices)/len(prices):.2f}")

            st.divider()

            for r in results:
                badge_cls = f"badge-{r['source'].lower()}"
                source_label = r['source'].capitalize()
                sim_pct = int(r['score'] * 100)

                with st.container():
                    c_img, c_info = st.columns([1, 3])

                    with c_img:
                        if r.get("image_url"):
                            try:
                                st.image(r["image_url"], use_container_width=True)
                            except Exception:
                                st.markdown("🖼️")

                    with c_info:
                        st.markdown(
                            f'<span class="source-badge {badge_cls}">{source_label}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**{r['name']}**")
                        if r.get("brand"):
                            st.caption(r["brand"])

                        col_p, col_s = st.columns(2)
                        with col_p:
                            st.markdown(
                                f'<div class="price-tag">{r["price"] or "N/A"}</div>',
                                unsafe_allow_html=True,
                            )
                        with col_s:
                            st.markdown(f"Similarity: **{sim_pct}%**")
                            st.markdown(
                                f'<div class="similarity-bar" style="width:{sim_pct}%"></div>',
                                unsafe_allow_html=True,
                            )

                        if r.get("rating"):
                            st.caption(f"⭐ {r['rating']}  ({r.get('reviews', '?')} reviews)")

                        if r.get("url"):
                            st.markdown(
                                f'<a class="buy-link" href="{r["url"]}" target="_blank">View on {source_label} →</a>',
                                unsafe_allow_html=True,
                            )

                    st.divider()
        elif results is not None:
            st.info("No results found. Try a different image or query.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#c9848f;">
          <div style="font-size:4rem;">💄</div>
          <p style="font-size:1.1rem;margin-top:1rem;">
            Upload a product photo or type a description to find matching products
          </p>
        </div>
        """, unsafe_allow_html=True)
