import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Streamlit Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Advanced Seaborn Dashboard 🎬", layout="wide")

st.title("🎨 Advanced Seaborn Dashboard")
st.write("Interactive analytics using Seaborn, Matplotlib & Streamlit")

sns.set_style('darkgrid')


# ---------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\ADMIN\vs code projects\movie_1.csv")  

movie = load_data()   # 559 rows × 6 columns


# ---------------------------------------------------------------
# Sidebar Filters (Genre + Column Filters)
# ---------------------------------------------------------------
st.sidebar.header("Filter Your Data")

# Genre filter
genres = movie['Genre'].unique()
selected_genres = st.sidebar.multiselect(
    "Filter by Genre:",
    options=genres,
    default=list(genres)
)

# Column filters – dynamic
numeric_cols = movie.select_dtypes(include=['int64','float64']).columns.tolist()

selected_min_max = {}
for col in numeric_cols:
    min_val = float(movie[col].min())
    max_val = float(movie[col].max())
    selected_min_max[col] = st.sidebar.slider(
        f"{col} Range:",
        min_val, max_val, (min_val, max_val)
    )

# Apply all filters
filtered_movie = movie[movie['Genre'].isin(selected_genres)]

for col, (low, high) in selected_min_max.items():
    filtered_movie = filtered_movie[(filtered_movie[col] >= low) & (filtered_movie[col] <= high)]


# ---------------------------------------------------------------
# Tabs (Dynamic Plots + Heatmap + Pairplot + Dataset Overview)
# ---------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Choose Plot", "🔥 Heatmap", "📈 Pairplot", "📄 Dataset Overview"]
)


# ---------------------------------------------------------------
# TAB 1 – Choose Plot
# ---------------------------------------------------------------
with tab1:
    st.header("Plot Selection")

    plot_type = st.selectbox(
        "Choose Plot Type:",
        [
            "LM Plot",
            "KDE Plot",
            "Hist Plot",
            "Jointplot – Hex",
            "Jointplot – Reg",
            "Jointplot – Resid",
            "Jointplot – KDE",
            "Jointplot – Scatter",
            "Distplot"
        ]
    )

    # ---------- LM Plot ----------
    if plot_type == "LM Plot":
        fig = sns.lmplot(
            data=filtered_movie,
            x="CriticRating",
            y="AudienceRating",
            hue="Genre",
            fit_reg=True,
            height=6,
            aspect=1.2
        )
        st.pyplot(fig)

    # ---------- KDE ----------
    elif plot_type == "KDE Plot":
        fig = plt.figure(figsize=(7,5))
        sns.kdeplot(
            data=filtered_movie,
            x="CriticRating",
            y="AudienceRating",
            fill=True,
            cmap="viridis"
        )
        st.pyplot(fig)

    # ---------- Hist Plot ----------
    elif plot_type == "Hist Plot":
        fig = plt.figure(figsize=(7,5))
        sns.histplot(filtered_movie['AudienceRating'], kde=True)
        st.pyplot(fig)

    # ---------- JOINTPLOTS ----------
    elif "Jointplot" in plot_type:
        joint_kind = plot_type.split("–")[1].strip().lower()   # hex, reg, resid, kde, scatter

        fig = sns.jointplot(
            data=filtered_movie,
            x="CriticRating",
            y="AudienceRating",
            kind=joint_kind
        )
        st.pyplot(fig)

    # ---------- Distplot ----------
    elif plot_type == "Distplot":
        fig = plt.figure(figsize=(7,5))
        sns.kdeplot(filtered_movie['CriticRating'], fill=True)
        st.pyplot(fig)

    st.success("Plot Generated Successfully!")


# ---------------------------------------------------------------
# TAB 2 – Heatmap
# ---------------------------------------------------------------
with tab2:
    st.header("Correlation Heatmap")

    corr = filtered_movie.corr(numeric_only=True)
    fig = plt.figure(figsize=(8,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot(fig)


# ---------------------------------------------------------------
# TAB 3 – Pairplot
# ---------------------------------------------------------------
with tab3:
    st.header("Pairplot Viewer")

    if st.checkbox("Generate Pairplot (slow)"):
        fig = sns.pairplot(
            filtered_movie[['CriticRating','AudienceRating','BudgetMillion','Genre']],
            hue='Genre'
        )
        st.pyplot(fig)
        st.success("Pairplot Generated!")


# ---------------------------------------------------------------
# TAB 4 – Dataset Overview (559 rows × 6 columns)
# ---------------------------------------------------------------
with tab4:
    st.header("Dataset Overview")

    st.subheader("First 10 Rows")
    st.dataframe(filtered_movie.head(10))

    st.subheader("Shape")
    st.write(f"Rows: {filtered_movie.shape[0]}")
    st.write(f"Columns: {filtered_movie.shape[1]}")

    st.subheader("Statistics")
    st.write(filtered_movie.describe())

    st.info("Dataset: 559 rows × 6 columns loaded successfully.")
