import streamlit as st

# Tab implementation
def tab_season(mytab):
    with mytab:
        st.title("Season Overview")

        season_summary = st.session_state['season_summary']
        # Season analysis
        st.subheader("Season Summary")
        season_options = season_summary.keys()
        selected_season = st.selectbox(
            "Select a Season",
            season_options,
            key="season_selector"
        )
        if selected_season is None:
            st.write('No season selected')
        else:
            season_data = season_summary[selected_season]
            with st.expander(f"Details for Season {selected_season}", expanded=True):
                st.write(season_data)