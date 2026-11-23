"""
Backwards-compatible entry point.
Trains models and saves them under models/, then reminds how to launch the app.
"""

from train_models import train_and_save

if __name__ == "__main__":
    train_and_save()
    print("\nNext: streamlit run app.py")
    print("Or runtime-train demo: streamlit run ML_Final_project.py")
