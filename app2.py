import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
from prediction import load_model,predict


# Set page configuration
st.set_page_config(
    page_title="Retina Image Classifier",
    page_icon="👁️",
    layout="wide"
)

# Define the categories and their descriptions
categories = ['Central Serous Chorioretinopathy_Color Fundus',
 'Diabetic Retinopathy',
 'Disc Edema',
 'Glaucoma',
 'Healthy',
 'Macular Scar',
 'Myopia',
 'Pterygium',
 'Retinal Detachment',
 'Retinitis Pigmentosa']

category_descriptions = {
    'Central Serous Chorioretinopathy_Color Fundus': 
        "A condition where fluid builds up under the retina, causing a localized detachment. "
        "Symptoms include blurred or distorted vision, reduced visual acuity, and a dark spot in the center of vision. "
        "Often affects people under stress and can resolve on its own within 3-6 months.",
    
    'Diabetic Retinopathy': 
        "A diabetes complication that affects the eyes by damaging blood vessels in the retina. "
        "Symptoms may include floating spots, blurred vision, and eventually blindness if untreated. "
        "Regular eye exams are crucial for early detection and treatment.",
    
    'Disc Edema': 
        "Swelling of the optic disc, which may indicate increased intracranial pressure or inflammation. "
        "Symptoms can include headaches, nausea, and vision changes. "
        "Requires prompt medical attention as it can be a sign of serious underlying conditions.",
    
    'Glaucoma': 
        "A group of eye conditions that damage the optic nerve, often due to abnormally high pressure in the eye. "
        "Initially symptomless but can lead to gradual vision loss starting with peripheral vision. "
        "Early treatment can help prevent vision loss.",
    
    'Healthy': 
        "A normal, healthy retina with no pathological findings. "
        "Regular eye check-ups are still recommended to monitor eye health.",
    
    'Macular Scar': 
        "Scar tissue in the macula (central part of the retina), often resulting from previous inflammation or injury. "
        "Can cause permanent central vision loss or distortion. "
        "Treatment focuses on preventing further damage.",
    
    'Myopia': 
        "Commonly known as nearsightedness, a condition where distant objects appear blurry. "
        "Caused by an elongated eyeball or overly curved cornea. "
        "Correctable with glasses, contact lenses, or surgery.",
    
    'Retinal Detachment': 
        "A serious condition where the retina pulls away from its supporting tissue. "
        "Symptoms include sudden flashes of light, floaters, and a curtain-like shadow over vision. "
        "Requires immediate medical attention to prevent permanent vision loss.",
    
    'Retinitis Pigmentosa': 
        "A group of rare genetic disorders that involve breakdown and loss of cells in the retina. "
        "Symptoms begin with night blindness and progress to tunnel vision. "
        "No cure exists, but treatment can help manage symptoms.",
    'Pterygium':
            "Pterygium, often called surfer's eye, is a non-cancerous growth of fleshy tissue on the conjunctiva, the clear membrane covering the white part of the eye. It typically starts near the corner of the eye and may extend toward the cornea." 
            " Long-term exposure to UV light, wind, dust, and dry conditions are common causes."
        }





# Function to display results
def display_results(probabilities, image, category_descriptions):
    # Find the most likely category
    max_prob_index = np.argmax(probabilities)
    max_category = categories[max_prob_index]
    max_probability = probabilities[max_prob_index]
    
    # Display the uploaded image and prediction
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Uploaded Retina Image")
        st.image(image, use_container_width=True)
        
        st.subheader("Diagnosis")
        st.markdown(f"**Most likely condition:** {max_category}")
        st.markdown(f"**Confidence:** {max_probability:.2%}")
        
        st.subheader("Description")
        st.markdown(category_descriptions[max_category])
        
        st.markdown("### Next Steps")
        if max_category != "Healthy":
            st.markdown("""
            * Consult with an ophthalmologist for a comprehensive eye examination
            * Bring this report to your appointment for reference
            * Follow up with any recommended specialist referrals
            """)
        else:
            st.markdown("""
            * Continue with regular eye check-ups
            * Maintain eye health through proper diet and eye protection
            * Report any changes in vision promptly
            """)
    
    with col2:
        st.subheader("Probability Distribution")
        # Create a horizontal bar chart
        fig, ax = plt.figure(figsize=(10, 8)), plt.subplot()
        
        # Sort probabilities for better visualization
        sorted_indices = np.argsort(probabilities)
        sorted_probs = probabilities[sorted_indices]
        sorted_cats = [categories[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        bars = ax.barh(sorted_cats, sorted_probs, color='skyblue')
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1%}', va='center')
        
        # Highlight the highest probability
        if sorted_indices[-1] == max_prob_index:
            bars[-1].set_color('forestgreen')
        
        plt.xlabel('Probability')
        plt.title('Condition Probabilities')
        plt.tight_layout()
        
        # Display the plot in Streamlit
        st.pyplot(fig)
        
        # Display information about other possible conditions
        st.subheader("Other Possible Conditions")
        # Get the top 3 conditions excluding the max
        other_indices = np.argsort(probabilities)[-4:-1][::-1]
        for idx in other_indices:
            with st.expander(f"{categories[idx]} ({probabilities[idx]:.1%})"):
                st.write(category_descriptions[categories[idx]])

# Information section
def show_info():
    with st.expander("ℹ️ About Eye Conditions", expanded=False):
        st.markdown("""
        ## Common Eye Conditions
        
        This app helps identify nine common retinal conditions. Below is brief information about each:
        """)
        
        for cat in categories:
            st.markdown(f"### {cat}")
            st.markdown(category_descriptions[cat])
            st.markdown("---")

# Main app
def main():
    # App title and description
    st.title("👁️ RetinaScan: Quick Eye Condition Detector")
    st.markdown("""
    ### Upload a retina image for quick assessment
    
    This tool helps **quickly identify potential eye conditions** from retinal images. It's designed to:
    - Assist healthcare providers in preliminary assessments
    - Help patients understand possible conditions before consultation
    - Provide immediate information about various eye conditions
    
    > **Note:** This tool is for informational purposes only and does not replace professional medical diagnosis.
    """)
    
    # Load model at app startup
    model = load_model()
    
    # Information section
    show_info()
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a retina image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display the image
        image = Image.open(uploaded_file)
        
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        # Show processing message
        with st.spinner('Analyzing retina image...'):
            # Get predictions
            probabilities = predict(model, image)
            
            # Display results
            display_results(probabilities, image, category_descriptions)
            
            st.markdown("---")
           
            
            st.caption("""
            **Disclaimer:** This application is intended for educational and informational purposes only. 
            It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
            Always seek the advice of your physician or other qualified health provider with any questions 
            you may have regarding a medical condition.
            """)

if __name__ == "__main__":
    main()
