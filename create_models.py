#!/usr/bin/env python
"""Generate 4-feature model files for Flask app"""

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import mysql.connector

print("="*60)
print("GENERATING 4-FEATURE MODEL FOR FLASK")
print("="*60)

try:
    # Connect to database
    print("\n1. Connecting to database...")
    connection = mysql.connector.connect(
        host='18.136.157.135',
        user='dm_team',
        password='DM!$Team@&27920!',
        database='project_itsm'
    )
    print("✓ Database connected")

    # Load data
    print("\n2. Loading data...")
    query = 'select * from dataset_list'
    data_list = pd.read_sql_query(query, connection)
    print(f"✓ Data loaded: {data_list.shape[0]} rows")

    # Prepare features
    print("\n3. Preparing 4-feature model...")
    X_model = data_list.loc[:,['CI_Cat','CI_Subcat','WBS','Category']].copy()
    y_model = data_list['Priority'].copy()
    print(f"✓ Features: {list(X_model.columns)}")

    # Encode features
    print("\n4. Encoding categorical features...")
    enc_4feature = LabelEncoder()
    for i in range(4):
        X_model.iloc[:,i] = enc_4feature.fit_transform(X_model.iloc[:,i].astype(str))
    print("✓ Encoding complete")

    # Split data
    print("\n5. Splitting data (70% train, 30% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y_model, test_size=0.3, random_state=10
    )
    print(f"✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Scale features
    print("\n6. Scaling features...")
    sc_4feature = StandardScaler()
    X_train_scaled = sc_4feature.fit_transform(X_train)
    X_test_scaled = sc_4feature.transform(X_test)
    print("✓ Scaling complete")

    # Train model
    print("\n7. Training RandomForest (max_depth=27)...")
    model_4feature = RandomForestClassifier(
        max_depth=27, random_state=10, n_estimators=100
    )
    model_4feature.fit(X_train_scaled, y_train)
    test_accuracy = model_4feature.score(X_test_scaled, y_test)
    print(f"✓ Model trained (accuracy: {test_accuracy:.4f})")

    # Create encoders
    print("\n8. Creating priority/WBS encoders...")
    priority_enc = LabelEncoder()
    priority_enc.fit(y_model)
    
    wbs_enc = LabelEncoder()
    wbs_enc.fit(data_list['WBS'])
    print("✓ Encoders created")

    # Save all artifacts
    print("\n9. Saving model files...")
    joblib.dump(model_4feature, 'model1_rf.joblib')
    joblib.dump(sc_4feature, 'scaler.joblib')
    joblib.dump(priority_enc, 'label_encoder.pkl')
    joblib.dump(wbs_enc, 'wbs_encoder.pkl')
    print("✓ All files saved:")
    print("   - model1_rf.joblib")
    print("   - scaler.joblib")
    print("   - label_encoder.pkl")
    print("   - wbs_encoder.pkl")

    print("\n" + "="*60)
    print("✅ SUCCESS! Flask app is ready to use")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python app.py")
    print("2. Open: http://127.0.0.1:5000")
    print("="*60 + "\n")

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
