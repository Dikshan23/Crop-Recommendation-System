"""
Prediction history tracking module.
Handles saving and retrieving crop prediction history from Supabase.
"""

import streamlit as st
from datetime import datetime
from utils.supabase_client import supabase
from utils.auth import get_user_email, get_user_id


def save_prediction_to_history(user, n, p, k, temp, hum, ph, rain, predicted_crop, confidence=None):
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
        confidence: Optional confidence score (0.0-1.0)
    
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        user_email = get_user_email(user)
        user_id = get_user_id(user)

        if not user_email or user_email == "User":
            st.warning("Unable to save history: User email not found")
            # still proceed if we have a stable user_id (best-effort)
            if not user_id:
                return False
        
        # Prepare prediction data
        prediction_data = {
            # keep both fields for backward compatibility during migration
            "user_id": user_id,
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
        
        # Add confidence if provided and valid
        if confidence is not None and confidence >= 0:
            prediction_data["confidence"] = float(confidence)
        
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
        user_id = get_user_id(user)
        user_email = get_user_email(user)

        if not user_id and (not user_email or user_email == "User"):
            st.warning("Unable to retrieve history: User identifier not found")
            return []

        # Query predictions for this user, prefer user_id when available
        query = supabase.table("prediction_history").select("*")
        if user_id:
            query = query.eq("user_id", user_id)
        else:
            query = query.eq("user_email", user_email)

        query = query.order("created_at", desc=True)
        
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
        user_id = get_user_id(user)
        user_email = get_user_email(user)

        if not user_id and (not user_email or user_email == "User"):
            return 0

        query = supabase.table("prediction_history").select("id", count="exact")
        if user_id:
            query = query.eq("user_id", user_id)
        else:
            query = query.eq("user_email", user_email)

        response = query.execute()

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
