import time
import pandas as pd
import numpy as np

CITY_DATA = { 'chicago': 'chicago.csv',
              'new york city': 'new_york_city.csv',
              'washington': 'washington.csv' }

def get_filters():
    """
    Asks user to specify a city, month, and day to analyze.

    Returns:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    """
    print('Hello! Let\'s explore some US bikeshare data!')
    # TO DO: get user input for city (chicago, new york city, washington). HINT: Use a while loop to handle invalid inputs
    city = 'invalid'
    while city == 'invalid' :
        user_city = input("do you want to filter the dataset using a city ? type chicago or new york city or washington or all \n")
        if user_city == 'chicago'or user_city == 'new york city' or user_city == 'washington' or user_city == 'all':
            city = user_city
      

    # TO DO: get user input for month (all, january, february, ... , june)
    month = 'invalid'
    months = ['january','february','march','april','may','june','july','august','september','october','november','december', 'all']
    while month == 'invalid' :
        user_month = input("do you want to filter the dataset using a month ? type 'january','february','march','april','may','june','july','august','september','october','november','december' \n")
        if user_month in months and user_month != 'all':
            month = months.index(user_month) + 1
        elif user_month == 'all':
            month = user_month

    # TO DO: get user input for day of week (all, monday, tuesday, ... sunday)
    day = 'invalid'
    days = ['saturday', 'sunday','monday','tuesday', 'wednesday', 'thursday', 'friday', 'all']
    while day == 'invalid' :
        user_day = input("do you want to filter the dataset using a day ? type 'saturday', 'sunday','monday','tuesday', 'wednesday', 'thursday', 'friday', 'all' \n")
        if user_day in days:
            day = user_day.title()

    print('-'*40)
    return city, month, day

def load_data(city, month, day):
    """
    Loads data for the specified city and filters by month and day if applicable.

    Args:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    Returns:
        df - Pandas DataFrame containing city data filtered by month and day
    """
    df = pd.read_csv(CITY_DATA[city])
    
    # Start Time column to datetime
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    
    # extract month and day of week to create new columns
    df['month'] = df['Start Time'].dt.month
    df['day_of_week'] = df['Start Time'].dt.weekday_name
    
    if month != 'all':
        df = df[df['month'] == month]
    
    if day != 'All':
        df = df[df['day_of_week'] == day]

    return df

city, month, day = get_filters()

print(load_data(city, month, day).head())
#print(data.head())