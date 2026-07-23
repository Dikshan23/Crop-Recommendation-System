import pandas as pd
import streamlit as st

from admin.shared import get_admin_db, require_admin_page, show_admin_db_warning


def render():
    require_admin_page("View Report")

    st.title("📈 View Report")
    show_admin_db_warning()

    db = get_admin_db()

    try:
        users = db.table("profiles").select("id, fullname, email").execute().data
        records = (
            db.table("prediction_history")
            .select("*")
            .order("created_at", desc=True)
            .execute()
            .data
        )
    except Exception as exc:
        st.error(f"Unable to load report: {exc}")
        return

    df_users = pd.DataFrame(users)
    df_records = pd.DataFrame(records)

  
    # Summary Metrics
   
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Users", len(df_users))

    with col2:
        st.metric("Total Predictions", len(df_records))

    
    # Crop Distribution Chart
   
    if not df_records.empty and "predicted_crop" in df_records.columns:
        st.subheader("Crop Prediction Distribution")
        crop_count = df_records["predicted_crop"].value_counts()
        st.bar_chart(crop_count)
    else:
        st.info("No prediction records available yet.")

   
    # User Table
    
    st.subheader("Registered Users")

    if not df_users.empty:
        st.dataframe(
            df_users,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No users found.")

    # Prediction Records Table
   
    st.subheader("Prediction History")

    if not df_records.empty:
        st.dataframe(
            df_records,
            width = "stretch",
            hide_index=True,
        )
    else:
        st.info("No prediction records found.")