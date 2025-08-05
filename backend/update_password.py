#!/usr/bin/env python3

from sqlalchemy import create_engine
from models import User
from sqlalchemy.orm import sessionmaker
from jwt_ import get_password_hash

# Create engine and session
engine = create_engine('sqlite:///./botsonic.db')
Session = sessionmaker(bind=engine)
session = Session()

# Update user password
user = session.query(User).filter(User.email == "ranajawadriaz.work@gmail.com").first()
if user:
    # Set a known password
    user.hashed_password = get_password_hash("password123")
    session.commit()
    print(f"Updated password for user: {user.email}")
else:
    print("User not found")

session.close()
