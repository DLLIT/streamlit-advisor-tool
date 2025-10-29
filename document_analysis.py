import streamlit as st
from PIL import Image
import io
import os
from openai import OpenAI
from dotenv import load_dotenv
import base64

# --- CHANGE 1: Load environment variables ---
# It's good practice to call load_dotenv() at the top level.
load_dotenv()

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
    st.stop()

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# --- CHANGE 2: Simplified and more robust analysis function ---
def analyze_lease_document(file_bytes, mime_type):
    """
    Uses the OpenAI API to analyze a Massachusetts lease agreement.
    This version sends the image data directly to a vision-capable model.
    """

    # --- CHANGE 3: A detailed and specific prompt for the AI ---
    # This prompt guides the AI to act as a paralegal focused on Massachusetts law.
    prompt = (
        """
        You are an expert paralegal specializing in Massachusetts landlord-tenant law.
        Your task is to analyze the provided lease agreement and identify any clauses
        that are illegal or unenforceable under Massachusetts General Laws.

        Please carefully review the document and list any problematic clauses.
        For each illegal clause you find, provide the following:
        1.  **Clause Identification**: Quote the exact illegal phrase or sentence from the lease.
        2.  **Explanation**: Clearly explain in simple, non-legal language why the clause is illegal in Massachusetts.
        3.  **Legal Reference**: Cite the specific Massachusetts General Law (MGL) chapter and section that makes the clause illegal.

        Common illegal clauses to look for in Massachusetts include, but are not limited to:
        - Requiring a tenant to pay for all repairs, regardless of fault.
        - Waiving the tenant's right to a jury trial.
        - Requiring the tenant to pay for utility costs for common areas without a separate meter.
        - Imposing a penalty for late rent that exceeds what is allowed by law (MGL c. 186, s. 15B).
        - Allowing the landlord to enter the premises without reasonable notice.
        - Requiring the tenant to pay rent in advance for more than the first month and a security deposit.

        If you do not find any illegal clauses, please state that "Based on this analysis, no illegal clauses were identified in the document."

        Present your findings in a clear, organized format using markdown.
        """
    )

    try:
        # --- CHANGE 4: Use a more capable model and a more direct API call for images ---
        # "gpt-4o" is a powerful vision-capable model suitable for this task.
        # This approach encodes the image and sends it in a single API call,
        # which is simpler and more modern than the File API method for this use case.
        # Note: This implementation currently only supports image files. PDF processing would require
        # a separate library (like PyMuPDF) to first convert PDF pages to images.

        if not mime_type.startswith("image/"):
            return "This tool currently only supports image files (JPEG, PNG). Please upload an image of your lease."

        # Encode the image in base64
        base64_image = base64.b64encode(file_bytes).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-5-nano", # A real, powerful model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000 # Increase tokens to allow for detailed analysis
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        return None

def process_uploaded_file(uploaded_file):
    """
    Processes the uploaded file and returns its bytes, MIME type, and a preview image.
    """
    file_bytes = uploaded_file.getvalue()
    mime_type = uploaded_file.type
    preview_image = None

    try:
        # We need a seek(0) to reset the buffer before reading it with PIL
        uploaded_file.seek(0)
        preview_image = Image.open(uploaded_file)
    except Exception:
        # This will fail for non-image files like PDFs, which is handled in the analyze function.
        pass

    return file_bytes, mime_type, preview_image

def main():
    # --- CHANGE 5: Update the UI text to be specific to the tool's purpose ---
    st.title("Massachusetts Lease Agreement Analyzer")
    st.write("This tool helps identify potentially illegal clauses in a Massachusetts residential lease agreement that may be harmful to tenants.")
    st.write("Upload a clear image (JPEG/PNG) of your lease pages, or use your camera to capture them.")

    # --- CHANGE 6: Add a prominent legal disclaimer ---
    st.warning(
        """
        **Disclaimer:** This tool provides an automated analysis and does not constitute legal advice.
        The information provided may not be complete or accurate. You should consult with a qualified
        legal professional for advice on your specific situation.
        """
    )

    input_method = st.radio("Choose input method:", ("Upload an Image File", "Use Camera"))

    uploaded_file = None
    if input_method == "Upload an Image File":
        # Simplified file uploader to focus on images.
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Take a picture of the lease page")

    if uploaded_file is not None:
        file_bytes, mime_type, preview_image = process_uploaded_file(uploaded_file)

        if preview_image:
            st.image(preview_image, caption="Image to be Analyzed", use_column_width=True)

            with st.spinner("Analyzing your lease document... This may take a moment."):
                analysis_result = analyze_lease_document(file_bytes, mime_type)
                if analysis_result:
                    st.success("Analysis Complete")
                    st.markdown(analysis_result) # Use markdown to render the formatted response
        else:
            st.error("Could not process the uploaded file. Please ensure it is a valid image.")

# Standard Python entry point
if __name__ == "__main__":
    main()
