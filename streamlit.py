import streamlit as st
import PyPDF2
from io import BytesIO

# Set the title of the app
st.title('PDF Upload and Parsing App')

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.read()
    
    # Use BytesIO to handle the uploaded PDF file
    pdf_reader = PyPDF2.PdfFileReader(BytesIO(bytes_data))
    
    # Assuming we want to extract text from the first page
    page = pdf_reader.getPage(0)
    page_text = page.extractText()
    
    # Display the extracted text
    st.write("Text extracted from the first page of the PDF:")
    st.text(page_text)