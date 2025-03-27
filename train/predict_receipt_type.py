import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# testing the custom dataset to identify the receipt type
import pandas as pd
import random

# Define the expanded dataset with more entries per category
expanded_data = {
    "Transportation": [
        "Flight", "Airfare", "Taxi", "Rental Car", "Train", "Shuttle", "Bus", "Uber", "Lyft", "Parking", "Fuel",
        "Travel Insurance", "Baggage Fees", "Departure", "Arrival", "Boarding", "Boarding Pass", "Commuting", 
        "Ride", "Ticket", "Transit", "Train Station", "Subway", "Limo", "Car Service", "Carpool", "Charter", 
        "Airport", "Terminal", "Ticket Price", "Cancellation Fee", "Flight Delay", "Layover", "Mileage", "Car Hire",
        "Airport Shuttle", "Taxi Fare", "Public Transport", "Highway", "Car Rental", "SUV", "Luxury Car", "Bus Fare", 
        "Road Trip", "Gas", "Tolls", "Car Parking", "Parking Fee", "Driving", "Gasoline", "Petrol", "Travel Reimbursement",
        "Cross-country", "Train Ticket", "Round-trip", "One-way Ticket", "Flight Upgrade", "First Class", "Economy Class",
        "Frequent Flyer", "Check-in", "Flight Change", "Cancellation", "Refund", "Baggage Allowance", "Carry-on", "Luggage"
    ],
    
    "Accommodation": [
        "Hotel", "Lodging", "Room Service", "Check-in", "Check-out", "Reservation", "Suite", "Motel", "Bed & Breakfast", 
        "Airbnb", "Room Rate", "Booking", "Stay", "Accommodation Tax", "Cleaning Fee", "Vacation Rental", "Hostel", 
        "Inn", "Luxury Suite", "Hotel Booking", "Penthouse", "Spa", "Luxury Resort", "Airbnb Listing", "Guest House", 
        "Room Amenities", "Wi-Fi", "Refrigerator", "Concierge", "Conference Room", "Event Venue", "Pool", "Gym", 
        "Fitness Center", "Luggage Storage", "Hotel Service", "Business Center", "Room Upgrade", "Room Key", "Rates", 
        "Business Trip", "Hotel Fee", "Hotel Reservation", "Booking Confirmation", "Online Booking", "Room Reservation", 
        "Guest Review", "Hotel Review", "Resort Fee", "Night Stay", "Check-out Time", "Room Availability", "Booking.com", 
        "Expedia", "Hotel Wi-Fi", "Cleaning Service", "Stay Duration", "Hotel Loyalty Program", "Vacation Package", 
        "Tourist Accommodation", "Hotel Shuttle", "Hotel Lobby", "Pet-friendly", "Airbnb Experience", "Hotel Voucher"
    ],
    
    "Meals": [
        "Breakfast", "Lunch", "Dinner", "Snacks", "Coffee", "Tea", "Tips", "Gratuity", "Per Diem", "Meal Allowance", 
        "Restaurant", "Buffet", "Catering", "Bar", "Room Service", "Dining", "Food & Beverage", "Soda", "Juice", 
        "Smoothie", "Water", "Alcohol", "Wine", "Beer", "Cocktail", "Appetizers", "Starter", "Main Course", 
        "Side Dishes", "Dessert", "Salad", "Soup", "Beverage", "Lunchbox", "Brunch", "Takeout", "Fast Food", 
        "Restaurant Bill", "Dinner Reservation", "Food Delivery", "Food Truck", "Cafeteria", "Buffet Table", "Tasting Menu",
        "Dinner Party", "Gourmet", "Takeaway", "Home-cooked Meal", "Catering Service", "Room Dining", "Chef", "Specialty Food",
        "Dinner Buffet", "Alcoholic Drink", "Sushi", "Sandwich", "Pasta", "Pizza", "Steak", "Vegetarian", "Vegan", 
        "Dessert Menu", "Café", "Breakfast Buffet", "Pizza Delivery", "Grilled", "Barbecue", "Barista", "Dining Out", 
        "Food Photography", "Wine Pairing", "Meal Prep", "Meal Plan", "Diet", "Healthy Eating", "Calories", "Energy Drink"
    ],
    
    "Miscellaneous": [
        "Conference", "Business Meeting", "Event", "Wi-Fi", "Internet", "Phone Call", "Phone Charges", "Printing", 
        "Faxing", "Office Supplies", "Travel Reimbursement", "Tip", "Service Charge", "VAT", "Taxes", "Expense Report", 
        "Deductions", "Invoice", "Reimbursement", "Document", "Report", "File", "Legal Fees", "Business Expense", 
        "Conference Call", "Office Space", "Co-working", "Presentation", "Training", "Seminar", "Contract", "Agreement", 
        "Workshops", "Networking", "Networking Event", "Staff Meeting", "Team Building", "Employee Benefits", "Payroll", 
        "Salaries", "Business Consultant", "Service Fee", "Tax Filing", "Corporate Tax", "Corporate Account", "Contractor", 
        "HR", "Recruitment", "Payroll System", "Time Tracking", "Work Expenses", "Software License", "Office Rent", 
        "Work Laptop", "Employee Travel", "Team Lunch", "Video Conference", "Client Meeting", "Business Negotiation", 
        "B2B Meeting", "Consulting Fee", "Expense Category", "Report Filing", "Email Subscription", "Business Research", 
        "Professional Services", "Professional Membership", "Health Insurance", "Office Party", "Workplace Benefits", 
        "Company Car", "Work Performance"
    ],
    
    "Airline": [
        "Flight Ticket", "Seat Reservation", "Upgrade", "First Class", "Economy Class", "Business Class", "Layover", 
        "Flight Change", "Cancellation", "Refund", "Flight Change Fee", "Frequent Flyer", "Miles", "Points", 
        "Frequent Flyer Program", "Loyalty Points", "Airline Loyalty Program", "Ticket Price", "Airline Fee", 
        "Booking Confirmation", "Airport Check-in", "Online Check-in", "Boarding Pass", "Travel Insurance", "Flight Route", 
        "Airline Lounge", "Baggage Allowance", "Flight Schedule", "Departure Gate", "Arrival Gate", "In-flight Service", 
        "Cabin Crew", "First Aid", "Emergency Exit", "Airline Cabin", "Plane Ticket", "Layover Time", "Flight Number", 
        "Arrival Time", "Departure Time", "Flight Delay", "Flight Cancellation", "Ticket Refund", "Upgrade Options", 
        "Business Class Seat", "Economy Seat", "Luggage Claim", "Customs", "Passport Control", "Immigration", 
        "In-flight Entertainment", "Airline Booking", "Flight Rebooking", "Checked Baggage", "Baggage Claim", 
        "Duty-Free Shopping", "Aircraft", "Jet", "Jet Lag", "Airline Baggage", "Security Check", "Travel Itinerary"
    ],
    
    "Car Rental": [
        "Car Hire", "Vehicle", "Rental Agreement", "Rental Fee", "Insurance", "Drop-off Fee", "Pick-up Fee", "Fuel Charges", 
        "GPS", "Car Rental Service", "Insurance Deductible", "Mileage", "Vehicle Inspection", "Car Model", "SUV Rental", 
        "Luxury Car", "Convertible", "Compact Car", "Full-size Car", "Minivan", "Rental Car Booking", "Car Rental Insurance", 
        "Collision Damage Waiver", "Fuel Policy", "Car Rental Terms", "Online Booking", "Vehicle Rental Location", 
        "Rental Car Reservation", "Road Assistance", "Car Key", "Car Pickup", "Car Drop-off", "Car Return", "Booking Confirmation", 
        "Discount Codes", "Rental Car Payment", "Car Rental Fees", "Additional Driver", "Rental Car Payment", "GPS System", 
        "Child Seat", "Car Navigation", "Vehicle Delivery", "Parking Fee", "Insurance Coverage", "Rental Car Extras", 
        "Fuel Tank", "Long-term Rental", "Car Rental Voucher", "Refundable Deposit", "Car Rental Terms and Conditions", 
        "Free Cancellation", "Car Rental Locations", "Car Rental Agency", "Car Rental Price", "Insurance Coverage", 
        "Roadside Assistance", "One-way Rental", "Car Hire Options", "SUV Hire", "Convertible Hire", "Car Rental Desk"
    ],
    
    "Travel Insurance": [
        "Coverage", "Medical", "Claim", "Emergency Assistance", "Trip Cancellation", "Lost Luggage", "Insurance Policy", 
        "Deductible", "Premium", "Travel Protection", "Medical Evacuation", "Reimbursement", "Accident Coverage", "Travel Medical", 
        "Trip Interruption", "Insurance Claim", "Emergency Travel Assistance", "Baggage Coverage", "Flight Delay Insurance", 
        "Medical Coverage", "Exclusions", "Accidental Death", "Travel Insurance Quote", "Insurance Company", "Travel Plan", 
        "Repatriation", "Policy Exclusions", "Tourist Insurance", "International Insurance", "Travel Refund", "Insurance Terms", 
        "Travel Insurance Certificate", "Pre-existing Conditions", "Travel Delay", "Flight Cancellation Coverage", "Travel Risks", 
        "Comprehensive Travel Insurance", "Adventure Travel Insurance", "Family Insurance", "Single Trip Insurance", "Annual Travel Insurance", 
        "Travel Insurance Package", "Travel Insurance Benefits", "Emergency Coverage", "Hospital Expenses", "Trip Delay", 
        "Accident Assistance", "Lost Property", "Hospitalization", "Insurance Reimbursement", "Trip Protection", "Baggage Loss", 
        "Natural Disaster Coverage", "Trip Protection Plan", "Trip Cancellation Policy", "Travel Security"
    ]
}

# Convert to DataFrame
df = pd.DataFrame([(category, item) for category, items in expanded_data.items() for item in items], 
                            columns=['Category', 'Item'])

# Assign labels (assuming business-related categories are labeled as 'business')
df['Label'] = df['Category'].apply(lambda x: 'business' if x != 'Meals' else 'personal')

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
