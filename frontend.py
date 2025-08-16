import streamlit as st
import requests
import time
import pdfkit

st.set_page_config(page_title= 'AI powered PDF Summarizer', layout = 'wide')

st.markdown('''
        <style>
            body{
                background-color : #282c34; /*Darker background */
                color : #abb2bf;  /* Text color */
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            }
            .stTextInput>div>div>input{
                font-size : 16px;
                padding : 12px;
                border-radius: 8px;
                border: 1px solid #61afef;
                background-color: #3e4451;
                color : #d1d5db;
            }
            .stButton>button{
                background-color : #61afef;
                color : #ffffff;
                font-size : 18px;
                font-weigtht : bold;
                padding : 12px 28px;
                broder-radius : 8px;
                transition : all 0.3s;
            }
            .stButton>button:hover{
                background-color: #569cd6;
                transform : translateY(-2px);
            }
            .stMarkdown, .stSubheader{
                color: #e06c75;
                font-weight : bold;
            }
            .summary-section{
                background-color : #3e4451;
                padding: 20px;
                border-radius: 10px;
                margin-bottom : 20px;
                border-left: 5px, solid #61afef;
            }
            .section-title{
                color: #61afef;
                font-size: 24px;
                margin-bottom: 15px;
            }
            .section-content{
                color : #d1d5db;
                font-size: 16px;
                line-height:1.6;
            }
        </style
    ''', unsafe_allow_html= True)

st.title("AI Powered PDF Summarizer")

st.markdown("Extract and summarize research papers with AI Powered efficieny")

pdf_url = st.text_input("Enter the Arxiv PDF URL: ", 
                        placeholder= 'https://arxiv.org/pdf/2401.02385.pdf')

status_placeholder = st.empty()

def format_section(title, content):
    return f"""
    <div class = 'summary-section">
        <div class = 'section-title'>{title}</div>
        <div class = 'section-content'>{content}</div>
    </div>
    """


if st.button("Summarize PDF"):
    if pdf_url:
        with st.spinner("Processing... This may take a few minutes"):
            status_placeholder.info("Fetching and summarizing the document")
            try:
                response = requests.post(
                    'http://localhost:8000/summarize_arxiv/',
                    json = {"url": pdf_url},
                    timeout= 3600
                )
                if response.status_code == 200:
                    data = response.json()
                    if "error" in data:
                        status_placeholder.error(f"{data['error']}")
                    else:
                        summary = data.get("summary", "no summary generated")
                        status_placeholder.success("summary ready")

                        sections = summary.split('#')[1:]
                        for section in sections:
                            parts = section.split("\n", 1)
                            if len(parts) == 2:
                                title, content = parts
                                st.markdown(
                                    format_section(title.strip(), content.strip()),
                                    unsafe_allow_html= True
                                )
                        if st.button("Download PDF"):
                            pdf_file = "summary.pdf"
                            pdfkit.from_string(summary, pdf_file)
                            with open(pdf_file, "rb") as f:
                                st.download_button(
                                    label="Download PDF",
                                    data=f,
                                    file_name="summary.pdf",
                                    mime="application/pdf"
                                )
                else:
                    status_placeholder.error("Failed to process the PDF. Please check the URL and try again")
            except requests.exceptions.Timeout:
                status_placeholder.error("Request time out. Please try again later")

            except Exception as e:
                status_placeholder.error(f"An error occured: {str(e)}")

    else:
        status_placeholder.warning("Please enter a valid Arxiv PDF URL")

st.markdown("--")
st.markdown("""
### Notes:
- Processing typically takes 3-5 mins depending on pdf length.
- Only Arvix PDF URLs are supported
- The summary is structured into key sections for better readability
- You can download the summary as a markdown file.
            """)