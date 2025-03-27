import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# testing the custom dataset to identify the receipt type
data = {
    "Transportation": ["Flight", "Airfare", "Taxi", "Rental Car", "Train", "Shuttle", "Bus", "Uber", "Lyft", "Parking", "Fuel", "Travel Insurance", "Baggage Fees", "Departure", "Arrival", "Boarding", "Boarding Pass"],
    "Accommodation": ["Hotel", "Lodging", "Room Service", "Check-in", "Check-out", "Reservation", "Suite", "Motel", "Bed & Breakfast", "Airbnb", "Room Rate", "Booking", "Stay", "Accommodation Tax", "Cleaning Fee"],
    "Meals": ["Breakfast", "Lunch", "Dinner", "Snacks", "Coffee", "Tea", "Tips", "Gratuity", "Per Diem", "Meal Allowance", "Restaurant", "Buffet", "Catering", "Bar", "Room Service", "Dining", "Food & Beverage"],
    "Miscellaneous": ["Conference", "Business Meeting", "Event", "Wi-Fi", "Internet", "Phone Call", "Phone Charges", "Printing", "Faxing", "Office Supplies", "Travel Reimbursement", "Tip", "Service Charge", "VAT", "Taxes"],
    "Airline": ["Flight Ticket", "Seat Reservation", "Upgrade", "First Class", "Economy Class", "Business Class", "Layover", "Flight Change", "Cancellation", "Refund", "Flight Change Fee", "Frequent Flyer", "Miles", "Points"],
    "Car Rental": ["Car Hire", "Vehicle", "Rental Agreement", "Rental Fee", "Insurance", "Drop-off Fee", "Pick-up Fee", "Fuel Charges", "GPS", "Car Rental Service", "Insurance Deductible", "Mileage"],
    "Travel Insurance": ["Coverage", "Medical", "Claim", "Emergency Assistance", "Trip Cancellation", "Lost Luggage", "Insurance Policy", "Deductible", "Premium", "Travel Protection"],
    "Taxation": ["Tax", "VAT", "Service Charge", "Local Tax", "Travel Tax", "Invoice", "Receipt", "Taxable Amount", "Refund", "Expense Report", "Deductions"]
}

df = pd.DataFrame([(category, item) for category, items in data.items() for item in items], columns=['Category', 'Item'])

df['Label'] = df['Category'].apply(lambda x: 'business' if x in ['Transportation', 'Accommodation', 'Meals', 'Miscellaneous', 'Airline', 'Car Rental', 'Travel Insurance', 'Taxation'] else 'personal')

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Item'])
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, 'svm_model.pkl')  # Saving the SVM model
joblib.dump(vectorizer, 'vectorizer.pkl')  # Saving the vectorizer
print("Model and vectorizer saved successfully.")
