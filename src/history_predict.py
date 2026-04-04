"""
Prediction history tracking module.
Handles saving and retrieving crop prediction history from Supabase.
"""

import streamlit as st
from datetime import datetime
from utils.supabase_client import supabase
from utils.auth import get_user_email


def save_prediction_to_history(user, n, p, k, temp, hum, ph, rain, predicted_crop):
    """
    Save a crop prediction to Supabase history table.
    
    Args:
        user: Supabase user object
        n: Nitrogen value
        p: Phosphorus value
        k: Potassium value
        temp: Temperature value
        hum: Humidity value
        ph: pH value
        rain: Rainfall value
        predicted_crop: The predicted crop name
    
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        user_email = get_user_email(user)
        
        if not user_email or user_email == "User":
            st.warning("Unable to save history: User email not found")
            return False
        
        # Prepare prediction data
        prediction_data = {
            "user_email": user_email,
            "nitrogen": float(n),
            "phosphorus": float(p),
            "potassium": float(k),
            "temperature": float(temp),
            "humidity": float(hum),
            "ph": float(ph),
            "rainfall": float(rain),
            "predicted_crop": str(predicted_crop).lower(),
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Insert into Supabase
        response = supabase.table("prediction_history").insert(prediction_data).execute()
        
        if response.data:
            return True
        else:
            st.error("Error saving prediction to history")
            return False
            
    except Exception as e:
        st.error(f"Failed to save prediction history: {str(e)}")
        return False


def get_user_prediction_history(user, limit=None):
    """
    Fetch all prediction history for the logged-in user from Supabase.
    
    Args:
        user: Supabase user object
        limit: Maximum number of records to fetch (None for all)
    
    Returns:
        list: List of prediction records sorted by newest first, or empty list if error
    """
    try:
        user_email = get_user_email(user)
        
        if not user_email or user_email == "User":
            st.warning("Unable to retrieve history: User email not found")
            return []
        
        # Query predictions for this user, ordered by newest first
        query = supabase.table("prediction_history").select("*").eq(
            "user_email", user_email
        ).order("created_at", desc=True)
        
        if limit:
            query = query.limit(limit)
        
        response = query.execute()
        
        return response.data if response.data else []
        
    except Exception as e:
        st.error(f"Failed to retrieve prediction history: {str(e)}")
        return []


def get_prediction_count(user):
    """
    Get the total count of predictions for the logged-in user.
    
    Args:
        user: Supabase user object
    
    Returns:
        int: Number of predictions made by user
    """
    try:
        user_email = get_user_email(user)
        
        if not user_email or user_email == "User":
            return 0
        
        response = supabase.table("prediction_history").select(
            "id", count="exact"
        ).eq("user_email", user_email).execute()
        
        return response.count if response.count else 0
        
    except Exception as e:
        return 0


def delete_prediction(prediction_id):
    """
    Delete a single prediction record from history.
    
    Args:
        prediction_id: ID of the prediction to delete
    
    Returns:
        bool: True if deleted successfully, False otherwise
    """
    try:
        response = supabase.table("prediction_history").delete().eq(
            "id", prediction_id
        ).execute()
        
        return True
        
    except Exception as e:
        st.error(f"Failed to delete prediction: {str(e)}")
        return False
