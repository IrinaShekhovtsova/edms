import csv
import datetime
import pandas as pd

MONTHS_MAP = {
    'Январь': 1,
    'Февраль': 2,
    'Март': 3,
    'Апрель': 4,
    'Май': 5,
    'Июнь': 6,
    'Июль': 7,
    'Август': 8,
    'Сентябрь': 9,
    'Октябрь': 10,
    'Ноябрь': 11,
    'Декабрь': 12,
}

def get_non_working_dates(filename='calendar.csv'):
    non_working_dates = set()
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            year = int(row['Год/Месяц'])
            for month_name in MONTHS_MAP:
                days_str = row.get(month_name)
                if days_str:
                    month = MONTHS_MAP[month_name]
                    days = days_str.split(',')
                    for day in days:
                        day = day.strip()
                        if '*' in day or not day:
                            continue
                        try:
                            day_int = int(day)
                            date = datetime.date(year, month, day_int)
                            non_working_dates.add(date)
                        except ValueError:
                            continue
    return non_working_dates

def calculate_working_hours(start_datetime, end_datetime, non_working_dates):
    work_start_time = datetime.time(9, 0, 0)
    work_end_time = datetime.time(18, 0, 0)
    total_working_hours = datetime.timedelta()

    current_datetime = start_datetime

    while current_datetime < end_datetime:
        current_date = current_datetime.date()

        if (current_date not in non_working_dates):
            day_start = max(current_datetime, datetime.datetime.combine(current_date, work_start_time))
            day_end = min(end_datetime, datetime.datetime.combine(current_date, work_end_time))

            if day_start < day_end:
                total_working_hours += day_end - day_start

        current_datetime = datetime.datetime.combine(current_date + datetime.timedelta(days=1), work_start_time)

    total_hours = total_working_hours.total_seconds() / 3600
    return total_hours


