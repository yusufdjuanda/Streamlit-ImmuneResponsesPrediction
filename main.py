import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
import pickle
import os
import numpy as np

def get_data():
    df1 = pd.read_parquet('./data/imm_encoded.parquet')
    df2 = pd.read_csv('./data/imm_original.tsv', sep = '\t')
    return df1, df2

def create_input_form(data1):

    import streamlit as st
    input_data = {}

    st.sidebar.header("Nano Particle's characteristics")

    # Categorical data
    np_types_options = ["Carbon", "Macromolecular Compound", "Oxide", "Salt"]
    shape_options = ["Hollow", "Dim-0", "Dim-1", "Dim-2"]
    surface_func_options = ["S.P.", "S.N."]
    animal_options = ["Mice", "Rat"]
    gender_options = ["male", "female"]
    method_options = ['OPA','IH.','IN.N.','IN.T.','I.T.']

    # Create the dropdowns in the sidebar
    input_data['NP Type'] = st.sidebar.selectbox("NP Type", options=np_types_options)
    input_data['Shape'] = st.sidebar.selectbox("Shape", options=shape_options)
    input_data['Surface Functionalization'] = st.sidebar.selectbox("Surface Functionalization", options=surface_func_options)
    input_data['Animal'] = st.sidebar.selectbox("Animal", options=animal_options)
    input_data['Gender'] = st.sidebar.selectbox("Gender", options=gender_options)
    input_data['Method'] = st.sidebar.selectbox("Method", options=method_options)


    #Numerical data
    slider_labels = [('Atomic mass (Com-1)', 'Com-1'), ('Atomic mass (Com-2)', 'Com-2'), ('Diameter','D'), ('Length','L'), 
                     ('Zeta','Zeta'), ('Specific Surface Area','SSA'), ('Molecular Weight','M.W'), ('Exposure Duration','E.D'), ('Exposure Frequency','E.T'),
                     ('Recover Duration (Days)','R.D'), ('Dose','Dose'), ('M.A', 'M.A'), ('M.W','M.W')]


    for label, col in slider_labels:
        input_data[col] = st.sidebar.slider(
            label, float(data1[col].min()), float(
                data1[col].max()), float(data1[col].mean())
        )



    return input_data 

def get_features(input_features):
    # Reference DataFrame columns
    df_columns = ['Carbon', 'Macromolecular Compound', 'Com-1', 'Com-2', 'Oxide', 'Salt', 'Dim-0', 'Dim-1', 'Dim-2',
              'Hollow', 'D', 'L', 'S.+', 'S.-', 'Zeta', 'SSA', 'Rat', 'Mice', 'male', 'female', 'M.A', 'M.W', 'OPA',
              'IH.', 'IN.N.', 'IN.T.', 'I.T.', 'E.D', 'E.T', 'R.D', 'Dose']

    # Initialize a dataframe with all zeros
    df_encoded = pd.DataFrame(0, index=[0], columns=df_columns)

    # input_raw = create_input_form(get_data())
    # For each item in the dictionary, update the corresponding value in the dataframe
    for key, value in input_features.items():
        if isinstance(value, str):
            # For string values, check if the column exists
            if value in df_encoded.columns:
                df_encoded[value] = 1
        elif key in df_encoded.columns:
            # For numeric values, directly assign the value to the corresponding column
            df_encoded[key] = value
    
    df_features = pd.DataFrame(df_encoded)

    return df_features

def get_prediction(input_data, model_dir = './models'):
    import os
    import pickle
    import shap

    prediction_dfs = []  # This will store all our prediction DataFrames

    # Iterate through all the model files in the directory
    for filename in os.listdir(model_dir):
        if filename.endswith(".pkl"):
            # Load the model from the file
            with open(os.path.join(model_dir, filename), 'rb') as file:
                model = pickle.load(file)

            # The key is the part of the filename before '_'
            key = filename.split('_')[1][:-4]
            
            # Predict the result and store it in the dictionary
            predictions = {key: model.predict(input_data)[0]}
            
            # Convert the dictionary to a DataFrame and append to our list
            prediction_dfs.append(pd.DataFrame([predictions]))

            df_predictions = pd.concat(prediction_dfs, axis=1)

    # Concatenate all prediction DataFrames into a single DataFrame and return
    return df_predictions

def create_radar_chart(input_data):

    import plotly.graph_objects as go

    # Create a figure
    fig = go.Figure()
    input_data = input_data.T
    # Add a trace for the input data
    fig.add_trace(go.Scatterpolar(
        r=input_data.values.flatten().tolist(),
        theta=input_data.index,
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 1]
            )
        ),
        showlegend=False
    )

    return fig

def categorize_based_on_reference(df_reference, df_to_categorize):
    import numpy as np

    df_reference['average'] = df_reference.iloc[:,31:].mean(axis=1)
    df_to_categorize['average'] = df_to_categorize.mean(axis=1)
    # Calculate quantiles from the reference dataframe
    quantiles = df_reference['average'].quantile([0.33, 0.67]).values

    # Define a function to apply to the dataframe to categorize data
    def categorize(value):
        if value <= quantiles[0]:
            return 'low'
        elif value <= quantiles[1]:
            return 'medium'
        else:
            return 'high'

    # Apply the function to the dataframe to categorize
    category = df_to_categorize['average'].apply(categorize)
    return category.values[0]

def display_predictions(category):

    st.subheader('Immune response prediction')
    st.write("The particular orgranism is predicted to have:")


    st.write(f"<span class='diagnosis bright-green'>{category.capitalize()} Immune Response </span>",
                 unsafe_allow_html=True)


    # st.write("Probability of being benign: ",
    #          model.predict_proba(input_data_scaled)[0][0])
    # st.write("Probability of being malignant: ",
    #          model.predict_proba(input_data_scaled)[0][1])

    # st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def st_shap(plot, *args, **kwargs):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot(*args, **kwargs).data}</body>"
    st.components.v1.html(shap_html, height=500)


def load_and_interpret_model(input_data, model_dir = './models'):
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))

    for i, filename in enumerate(model_files):
        # Load the model from the file
        with open(os.path.join(model_dir, filename), 'rb') as file:
            model = pickle.load(file)

        key = filename.split('_')[1][:-4]

        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)

        # get the names of the features
        feature_names = input_data.columns
        # get the SHAP values for the top 5 features
        top_indices = np.argsort(-np.sum(np.abs(shap_values.values), axis=0))[:5]

        row = i // 4
        col = i % 4

        # # SHAP dot plot
        # shap.summary_plot(shap_values.values[:, top_indices], feature_names=top_indices, plot_type="dot", 
        #                   color_bar_label='SHAP Value', show=False, plot_size=None)

        axs[row, col].barh(feature_names[top_indices], np.sum(np.abs(shap_values.values[:, top_indices]), axis=0))
        axs[row, col].invert_yaxis()  # To make the sequence descending
        axs[row, col].set_title(f'Model: {key}')
        axs[row, col].set_xlabel('')

    plt.tight_layout()
    st.pyplot(fig)


def playground():

    # load css
    with open("./assets/style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()),
                    unsafe_allow_html=True)

    with st.container():
        st.title("Immune Responses Prediction")
        st.write("""---""")
        st.write("""This webb app predicts the immune response of an organism based on the characteristics of the nano particle. 
                 The model is trained using Random Forest Regressor. The data and methods utilized are obtained from the paper **Yu et al., Sci. Adv. 2021; 7 : eabf4130**""")
                 
        # st.write(get_data().columns)

    data1, data2 = get_data()
    input_data = create_input_form(data1)
    features = get_features(input_data)
    prediction= get_prediction(features)
    st.subheader("Predicted Immune Components")
    st.table(prediction)
    category = categorize_based_on_reference(data1, prediction)



    col1, col2 = st.columns([4, 2])
    with col1:
        radar_chart = create_radar_chart(prediction)
        st.plotly_chart(radar_chart, use_container_width=True)

    with col2:
        # load the model
        display_predictions(category)

    if st.button('Interpret Model'):
        # st.write('Why hello there')
        load_and_interpret_model(features)

def run_model_performance():
    import streamlit as st
    st.header("Model Performance")
    st.write('---')
    st.write('The model is trained using Random Forest Regressor with 10-folShuffleSplit cross validation.')
    df = pd.read_csv('./data/eval_df.csv', sep = ',')
    df = df.drop([col for col in df.columns if 'Train' in col], axis=1)

    
    st.table(df)
    st.image('./image/rf_regression.png', use_column_width=True)

def main():
    import streamlit as st
    st.set_page_config(page_title="Immune Responses Prediction",
                       page_icon=":female-doctor:", layout="wide", initial_sidebar_state="expanded")
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Model Performance", "Playground"])

    if selection == "Model Performance":
        run_model_performance()
    elif selection == "Playground":
        playground()



if __name__ == '__main__':
    main()