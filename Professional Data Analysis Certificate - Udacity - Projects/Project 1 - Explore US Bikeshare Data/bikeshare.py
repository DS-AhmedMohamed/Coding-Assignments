import time
import pandas as pd
import numpy as np

CITY_DATA = { 'chicago': 'chicago.csv',
              'new york city': 'new_york_city.csv',
              'washington': 'washington.csv' }
months = ['january','february','march','april','may','june', 'all']

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
        user_city = input("do you want to filter the dataset using a city ? type chicago or new york city or washington or all \n").lower()
        if user_city in CITY_DATA.keys():
            city = user_city
        else:
            print("{} is an invalid input, please be sure to type of the available cities.".format(user_city))
      

    # TO DO: get user input for month (all, january, february, ... , june)
    month = 'invalid'
    while month == 'invalid' :
        user_month = input("do you want to filter the dataset using one of the first 6 months of the year? type 'january','february','march','april','may','june' or 'All' \n").lower()
        if user_month in months and user_month != 'all':
            month = months.index(user_month) + 1
        elif user_month == 'all':
            month = user_month
        else:
            print("{} is not a valid month, be sure to choose one of the first 6 months of the year only".format(user_month))

    # TO DO: get user input for day of week (all, monday, tuesday, ... sunday)
    day = 'invalid'
    days = ['saturday', 'sunday','monday','tuesday', 'wednesday', 'thursday', 'friday', 'all']
    while day == 'invalid' :
        user_day = input("do you want to filter the dataset using a day ? type 'saturday', 'sunday','monday','tuesday', 'wednesday', 'thursday', 'friday', 'all' \n").lower()
        if user_day in days:
            day = user_day.title()
        else:
            print("{} is not a valid day, try again.".format(user_day))

    print('-'*40)
    return city, month, day

#city, month, day = get_filters()

def load_data(city, month, day):
    """ this function takes city, month and day as arguments and filter the chosen dataset based on them. """
    
    df = pd.read_csv(CITY_DATA[city])
    
    # Start Time column to datetime
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    
    # extract month and day of week to create new columns
    df['month'] = df['Start Time'].dt.month
    df['day_of_week'] = df['Start Time'].dt.weekday_name
    df['start_hour'] = df['Start Time'].dt.hour
    
    if month != 'all':
        df = df[df['month'] == month]
    
    if day != 'All':
        df = df[df['day_of_week'] == day]

    return df


def time_stats(df):
    """Displays statistics on the most frequent times of travel."""

    print('\nCalculating The Most Frequent Times of Travel...\n')
    start_time = time.time()

    # TO DO: display the most common month
    common_month = df['month'].mode()[0]

    # TO DO: display the most common day of week
    common_day = df['day_of_week'].mode()[0]


    # TO DO: display the most common start hour
    common_hour = df['start_hour'].mode()[0]

    return print("the most common : hour = {}, day = {}, month = {}".format(common_hour, common_day, months[common_month-1]))

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def station_stats(df):
    """Displays statistics on the most popular stations and trip."""

    print('\nCalculating The Most Popular Stations and Trip...\n')
    start_time = time.time()

    # TO DO: display most commonly used start station
    common_start = df['Start Station'].mode()[0]

    # TO DO: display most commonly used end station
    common_end = df['End Station'].mode()[0]

    # TO DO: display most frequent combination of start station and end station trip
    common_comb = df.groupby(['Start Station','End Station']).size().idxmax()
    
    return print("the most common :\n # Start Station = {}\n # End Station = {}\n # Combination = {}.\n".format(common_start, common_end, common_comb))

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def trip_duration_stats(df):
    """Displays statistics on the total and average trip duration."""

    print('\nCalculating Trip Duration...\n')
    start_time = time.time()

    # TO DO: display total travel time in minutes
    total_travel_time = df['Trip Duration'].sum() / 60

    # TO DO: display mean travel time in minutes
    mean_travel_time = df['Trip Duration'].mean() / 60
    
    return print("Total Travel Time = {} mins , Mean Travel Time = {} mins".format(round(total_travel_time), round(mean_travel_time)))

#print("\nThis took %s seconds." % (time.time() - start_time))
#print('-'*40)


def user_stats(df):
    """Displays statistics on bikeshare users."""

    return print(df.describe())


# TO DO: Display counts of user types
def user_types(df) :
    data = df
    user_type_counts = data['User Type'].value_counts()
    
    return user_type_counts


# TO DO: Display counts of gender
def genders(df) :
    data = df
    if 'Gender' in data.columns:
        gender_counts = data['Gender'].value_counts()
        return gender_counts
    else:
        return "the dataset doesn't has a Gender column"
    
    

# TO DO: Display earliest, most recent, and most common year of birth
def earliest_YOB(df):
    data = df
    if 'Birth Year' in data.columns:
        return data['Birth Year'].min()
    else:
        return "the dataset hasn't a Birth Year column"


def mostrecent_YOB(df):
    data = df
    if 'Birth Year' in data.columns:
        return data['Birth Year'].max()
    else:
        return "the dataset hasn't a Birth Year column"

def mostcommon_YOB(df):
    data = df 
    if 'Birth Year' in data.columns:
        return data['Birth Year'].mode()[0]
    else:
        return "the dataset hasn't a Birth Year column"
    
def display_raw_data(df):
    check_input = "invalid"
    while check_input == "invalid":
        check_dataset = input("Do you want to check the dataset (1st 5 rows)? type yes or no\n").lower()
    
        if check_dataset == "yes":
            print(df.head())
            check_input = "valid"
        
            answer = "again"
            rows_num = 10
            while answer == "again":
                answer = input("Do you want to add the next 5 rows to the previous result? type 'yes' or 'no'\n").lower()
                if answer == 'yes':
                    print(df.head(rows_num))
                    rows_num += 5
                    answer = "again"
                elif answer == 'no':
                    print("Fine. Let's continue \n")
                else:
                    print("invalid answer, try again \n")
                    answer = "again"
                     
                    
                    
        elif check_dataset == "no":
            print("Fine, Lets continue.\n")
            print("-" * 50)
            check_input = "valid"
        else :
            print("{} is not a valid answer. try again.".format(check_dataset))
            print("-" * 50)


def main():
    while True:
        city, month, day = get_filters()
        df = load_data(city, month, day)
        
        print("Display Raw Data\n")
        display_raw_data(df)
            
                 
        time_stats(df)
        print("-" * 50)
                 
        station_stats(df)
        print("-" * 50)
        
        trip_duration_stats(df)
        print("-" * 50)
        
        print("user Stats : \n")
        user_stats(df)
        print("-" * 50)
        
        print("user types : \n")
        print(user_types(df))
        print("-" * 50)

        
        print("Genders : \n")
        print(genders(df))
        print("-" * 50)

        
        print("Earliest year of birth : {}".format(earliest_YOB(df)))
        print("-" * 50)

        print("Most recent year of birth : {}".format(mostrecent_YOB(df)))
        print("-" * 50)

        print("Most Common year of birth : {}".format(mostcommon_YOB(df)))
        print("-" * 50)

        valid_answer = ['yes','no']
        restart = input('\nWould you like to restart? Enter yes or no.\n')
        if restart.lower() == 'no':
            break
        elif restart.lower() not in valid_answer:
            print("your answer is not valid, the program will restart.\n")
            print("-" * 50)

if __name__ == "__main__":
	main()
