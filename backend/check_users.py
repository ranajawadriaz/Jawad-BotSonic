#!/usr/bin/env python3

from sqlalchemy import create_engine
from models import User
from sqlalchemy.orm import sessionmaker

# Create engine and session
engine = create_engine('sqlite:///./botsonic.db')
Session = sessionmaker(bind=engine)
session = Session()

# Query users
users = session.query(User).all()
print(f'Found {len(users)} users')

for user in users:
    print(f'User: {user.email}, Active: {user.is_active}')

session.close()
