import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

img_size = 100
st.header('Cotton Disease Prediction')

model = tf.keras.models.load_model('cotton_model.h5')

def prediction(x):
  array = model.predict(x) # getting prediction in form of probabilities
  classes = ['healthy','bacterial blight','curl virus','fusarium wilt'] # target variables
  array = list(np.ravel(array)) # converting the probability list to 1D
  prob = max(array) # fetching the maximum probability, as it indicates the correct class
  return classes[array.index(prob)] # through the index of max probability, getting the class name


uploaded_file = st.file_uploader("Upload cotton crop image")

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    show_img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(show_img, (img_size, img_size)) # resize image to 100 * 100
    img = np.array(img)
    img = img / 255 # normalize the image array
    img=img.reshape(-1, img_size, img_size, 3)

    st.image(show_img, channels="BGR") # show the uploaded image

    x = prediction(img)
    if x == 'healthy':
        st.markdown(f'The plant appears to be **{x}**.')

    elif x == 'fusarium wilt':
        st.markdown(f'The plant is diagnosed with **{x}**.')
        st.write('It is caused by the fungus Fusarium oxysporum f. sp. vasinfectum, is a major disease of cotton capable '
                'of causing significant economic loss. The fungus persists in soil as chlamydospores and in association with '
                'the roots of susceptible, resistant and non-cotton hosts as well as in seed.')
        st.write('[Read More](https://en.wikipedia.org/wiki/Fusarium_wilt) on wikipedia.')

    elif x == 'bacterial blight':
        st.markdown(f'The plant is diagnosed with **{x}**.')
        st.write('It is caused by Pseudomonas syringae pv. glycinea, which can also infect snap bean and lima bean. '
                'The pathogen overwinters in crop residue and can be seed transmitted.')
        st.write('[Read More](https://en.wikipedia.org/wiki/Bacterial_blight_of_cotton) on wikipedia.')

    elif x == 'curl virus':
        st.markdown(f'The plant is diagnosed with **{x}**.')
        st.write('It is caused by the Cotton leaf curl geminivirus (CLCuV). Leaves of infected cotton curl upward'
                ' and bear leaf-like enations on the underside along with vein thickening.')
        st.write('[Read More](https://en.wikipedia.org/wiki/Cotton_leaf_curl_virus) on wikipedia.')