# frontend.py
"""
Streamlit frontend for "⚖ The Legal Simplifier AI".

Run:
    streamlit run frontend.py

Requirements (example):
pip install streamlit langchain pdfplumber python-docx chromadb sentence-transformers transformers
pip install langchain-google-genai  # LangChain Google Gemini connector
"""

import os
import streamlit as st

from backend import (
    parse_file,
    chunk_and_store,
    summarize_text,
    analyze_document_for_risks,
    safe_analyze_document_for_risks,
    semantic_search,
    intelligent_date_extraction_and_validation,  # NEW
    semantic_search_with_intelligent_validation,  # NEW
    compare_documents,
    VECTORSTORE_PERSIST_DIR,
    simple_date_validator,
    semantic_search_with_dates,

)

st.set_page_config(page_title="⚖ The Legal Simplifier AI", layout="wide")

st.title("⚖ The Legal Simplifier AI")
st.write("An AI-powered tool to make legal documents understandable and actionable.")

# Ensure session_state defaults
if "doc1_text" not in st.session_state:
    st.session_state.doc1_text = ""
if "doc2_text" not in st.session_state:
    st.session_state.doc2_text = ""
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "collection_name" not in st.session_state:
    st.session_state.collection_name = "legal_docs"

# --- Section 1: Upload and Analyze Documents
st.header("📤 Upload & Analyze Documents")
uploaded_file = st.file_uploader("Upload primary document (.pdf or .docx)", type=["pdf", "docx"], key="u1")
if uploaded_file:
    with st.spinner("Parsing file..."):
        text = parse_file(uploaded_file)
        st.session_state.doc1_text = text
    st.success("File parsed and stored in memory.")
    st.text_area("Parsed text (primary)", value=st.session_state.doc1_text[:4000], height=200)

    # Auto-ingest into Chroma right after parsing
    with st.spinner("Chunking, embedding, and storing into Chroma vector store..."):
        chunk_and_store(
            text=st.session_state.doc1_text,
            collection_name=st.session_state.collection_name,
            chunk_size=500,
            chunk_overlap=50,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            persist_directory=VECTORSTORE_PERSIST_DIR,
        )
        st.session_state.vectorstore_ready = True
    st.success("Document ingested into vector store. Ready for semantic queries.")


# --- Section 2: Plain Language Summary
st.header("📑 Plain Language Summary")
if st.button("Generate Summary"):
    if not st.session_state.doc1_text:
        st.error("Please upload and parse a primary document first.")
    else:
        with st.spinner("Generating simplified summary..."):
            summary = summarize_text(st.session_state.doc1_text)
        st.info(summary)

# --- Section 3: Risk & Obligation Analysis
st.header("⚠️ Legal Risk & Obligation Analysis")
st.caption("Works with any legal document: contracts, licenses, agreements, leases, policies, etc.")

# Analysis options
col1, col2 = st.columns(2)
with col1:
    debug_mode = st.checkbox("Show debug information", value=False)
with col2:
    detailed_analysis = st.checkbox("Show detailed breakdown", value=True)

if st.button("🔍 Run Comprehensive Analysis", type="primary"):
    if not st.session_state.doc1_text:
        st.error("Please upload and parse a legal document first.")
    else:
        with st.spinner("Analyzing document for risks, obligations, and legal issues..."):
            try:
                # Show preview of text being analyzed
                if debug_mode:
                    st.write("**📄 Document Preview (first 500 characters):**")
                    st.code(st.session_state.doc1_text[:500] + "...")
                
                # Use the safer wrapper function
                risks, obligations = safe_analyze_document_for_risks(st.session_state.doc1_text)
                
                if debug_mode:
                    st.write("**🔧 Raw Analysis Results:**")
                    st.json({"risks": risks, "obligations": obligations})
                
                # === RISK ANALYSIS DISPLAY ===
                st.subheader("🚨 Risk Assessment")
                
                total_risks = sum(len(items) for items in risks.values())
                if total_risks > 0:
                    # Risk summary metrics
                    high_count = len(risks.get("High", []))
                    medium_count = len(risks.get("Medium", []))
                    low_count = len(risks.get("Low", []))
                    
                    # Display risk summary
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🔴 High Risk", high_count)
                    with col2:
                        st.metric("🟡 Medium Risk", medium_count) 
                    with col3:
                        st.metric("🟢 Low Risk", low_count)
                    with col4:
                        st.metric("📊 Total Risks", total_risks)
                    
                    # Display each risk category
                    for severity, items in risks.items():
                        if items:
                            if severity.lower() == "high":
                                st.error(f"**🔴 {severity.upper()} PRIORITY RISKS** ({len(items)} identified)")
                                if detailed_analysis:
                                    for i, item in enumerate(items, 1):
                                        st.error(f"**{i}.** {item}")
                                else:
                                    st.error("• " + "\n• ".join(items))
                                        
                            elif severity.lower() == "medium":
                                st.warning(f"**🟡 {severity.upper()} PRIORITY RISKS** ({len(items)} identified)")
                                if detailed_analysis:
                                    for i, item in enumerate(items, 1):
                                        st.warning(f"**{i}.** {item}")
                                else:
                                    st.warning("• " + "\n• ".join(items))
                                    
                            else:  # Low risk
                                st.info(f"**🟢 {severity.upper()} PRIORITY RISKS** ({len(items)} identified)")
                                if detailed_analysis:
                                    for i, item in enumerate(items, 1):
                                        st.info(f"**{i}.** {item}")
                                else:
                                    st.info("• " + "\n• ".join(items))
                else:
                    st.success("✅ **No significant risks identified** in this document")
                
                # === OBLIGATIONS ANALYSIS DISPLAY ===
                st.subheader("📋 Legal Obligations & Requirements")
                
                if obligations:
                    st.info(f"**Found {len(obligations)} obligation(s) that require attention:**")
                    
                    if detailed_analysis:
                        for i, obligation in enumerate(obligations, 1):
                            # Categorize obligations by type
                            obligation_lower = obligation.lower()
                            if any(word in obligation_lower for word in ['payment', 'pay', 'fee', 'cost', '$']):
                                st.warning(f"💰 **{i}. FINANCIAL:** {obligation}")
                            elif any(word in obligation_lower for word in ['report', 'submit', 'file', 'notify']):
                                st.info(f"📄 **{i}. REPORTING:** {obligation}")
                            elif any(word in obligation_lower for word in ['maintain', 'keep', 'preserve', 'update']):
                                st.info(f"🔧 **{i}. MAINTENANCE:** {obligation}")
                            elif any(word in obligation_lower for word in ['renew', 'extend', 'continue']):
                                st.warning(f"🔄 **{i}. RENEWAL:** {obligation}")
                            elif any(word in obligation_lower for word in ['comply', 'follow', 'adhere', 'regulation']):
                                st.error(f"⚖️ **{i}. COMPLIANCE:** {obligation}")
                            else:
                                st.write(f"📌 **{i}.** {obligation}")
                    else:
                        for obligation in obligations:
                            st.write(f"• {obligation}")
                else:
                    st.info("ℹ️ **No explicit obligations detected** in this document")
                
                # === ACTION ITEMS SUMMARY ===
                if risks.get("High") or obligations:
                    st.subheader("⚡ Immediate Action Items")
                    st.error("**URGENT - Requires immediate attention:**")
                    
                    action_items = []
                    
                    # Add high-risk items as action items
                    for high_risk in risks.get("High", []):
                        action_items.append(f"🚨 Address: {high_risk}")
                    
                    # Add time-sensitive obligations
                    for obligation in obligations:
                        if any(word in obligation.lower() for word in ['due', 'expire', 'deadline', 'immediate', 'urgent', 'overdue']):
                            action_items.append(f"⏰ Complete: {obligation}")
                    
                    if action_items:
                        for item in action_items:
                            st.write(f"• {item}")
                    else:
                        st.info("No immediate action items identified")
                
                # === DOCUMENT TYPE DETECTION ===
                if debug_mode:
                    st.subheader("📊 Document Analysis Details")
                    text_sample = st.session_state.doc1_text.lower()
                    
                    doc_types = []
                    if any(word in text_sample for word in ['license', 'licensing']):
                        doc_types.append("License/Permit")
                    if any(word in text_sample for word in ['lease', 'rent', 'tenant', 'landlord']):
                        doc_types.append("Lease Agreement")
                    if any(word in text_sample for word in ['employment', 'employee', 'employer', 'salary', 'wages']):
                        doc_types.append("Employment Contract")
                    if any(word in text_sample for word in ['service', 'services', 'provider', 'client']):
                        doc_types.append("Service Agreement")
                    if any(word in text_sample for word in ['purchase', 'sale', 'buy', 'sell', 'goods']):
                        doc_types.append("Purchase/Sale Agreement")
                    if any(word in text_sample for word in ['confidential', 'nda', 'non-disclosure']):
                        doc_types.append("Confidentiality Agreement")
                    if any(word in text_sample for word in ['loan', 'credit', 'financing', 'mortgage']):
                        doc_types.append("Financial Agreement")
                    
                    if doc_types:
                        st.info(f"**Detected Document Type(s):** {', '.join(doc_types)}")
                    else:
                        st.info("**Document Type:** Generic Legal Document")
                
            except Exception as e:
                st.error(f"❌ **Analysis Error:** {str(e)}")
                if debug_mode:
                    st.exception(e)
# --- Section: Enhanced Search with Date Validation (NEW)
st.header("🔍 Smart Validity Check")
st.caption("Ask about document validity - now with intelligent date analysis!")

validity_query = st.text_input("Ask about validity:", value="Is my insurance valid?", key="validity_search")

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Check Validity", type="primary"):
        if not st.session_state.vectorstore_ready:
            st.error("Please upload and process a document first.")
        else:
            with st.spinner("Analyzing document with date intelligence..."):
                try:
                    # Import the enhanced function (add this import to your backend imports)
                    from backend import semantic_search_with_dates
                    
                    answer = semantic_search_with_dates(
                        query=validity_query,
                        collection_name="legal_docs",
                        top_k=5
                    )
                    
                    # Display with appropriate styling based on content
                    if "🔴 NO" in answer or "NOT VALID" in answer:
                        st.error(answer)
                    elif "🟢 YES" in answer or "VALID" in answer:
                        st.success(answer)
                    elif "🟡" in answer or "EXPIRING" in answer:
                        st.warning(answer)
                    else:
                        st.info(answer)
                        
                except Exception as e:
                    st.error(f"Enhanced search failed: {str(e)}")
                    st.info("Try the regular semantic search above.")

with col2:
    if st.button("📅 Quick Date Check"):
        if not st.session_state.doc1_text:
            st.error("Please upload a document first.")
        else:
            with st.spinner("Extracting dates..."):
                try:
                    from backend import simple_date_validator
                    
                    results = simple_date_validator(st.session_state.doc1_text)
                    
                    if results["status"] == "EXPIRED":
                        st.error(f"❌ **EXPIRED**: {results['message']}")
                    elif results["status"] == "VALID":
                        st.success(f"✅ **VALID**: {results['message']}")
                    elif results["status"] == "EXPIRING_SOON":
                        st.warning(f"⚠️ **EXPIRING SOON**: {results['message']}")
                    else:
                        st.info(f"ℹ️ {results['message']}")
                    
                    if results["action_needed"]:
                        st.write("**Actions needed:**")
                        for action in results["action_needed"]:
                            st.write(f"• {action}")
                            
                except Exception as e:
                    st.error(f"Date check failed: {str(e)}")
# --- Section 4: Semantic Search
st.header("🔍 Semantic Search")
query_input = st.text_input("Enter your question", value="What are my obligations?")
if st.button("Search"):
    if not st.session_state.vectorstore_ready:
        st.error("Please ingest the primary document (chunk & store) before semantic searching.")
    else:
        with st.spinner("Running semantic search and generating final answer..."):
            answer = semantic_search(
                query=query_input,  # Fixed: was using undefined 'user_query' variable
                collection_name="legal_docs",
                top_k=5
                # model_name parameter is optional - will use GEMINI_MODEL from .env
            )
        st.info(answer)

# --- Section 5: Compare Documents
st.header("📄 Compare Documents (Redline Simulation)")
uploaded_file2 = st.file_uploader("Upload second document to compare (.pdf or .docx)", type=["pdf", "docx"], key="u2")
if uploaded_file2:
    with st.spinner("Parsing second file..."):
        st.session_state.doc2_text = parse_file(uploaded_file2)
    st.success("Second file parsed.")
    st.text_area("Parsed text (second)", value=st.session_state.doc2_text[:4000], height=200)

if st.button("Compare Documents"):
    if not st.session_state.doc1_text or not st.session_state.doc2_text:
        st.error("Please upload both primary and second documents to compare.")
    else:
        with st.spinner("Comparing documents..."):
            md = compare_documents(st.session_state.doc1_text, st.session_state.doc2_text)
        st.markdown(md)

# --- Download scaffolding (not wired to produce real files in this prototype)
st.header("Download")
st.write("Download functionality scaffolded (not wired to actual file outputs in this demo).")

st.caption("Backend persist directory: " + VECTORSTORE_PERSIST_DIR)