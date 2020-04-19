# import datetime, time
from datetime import datetime, time
from time import sleep

# wait 1 minute
def wait_1_min():
    wait = 1 * 60
    time.sleep(wait)
def wait_x_min(x=1):
    wait = x * 60
    time.sleep(wait)
    
def wait_1_hr():
    wait = 1 * 60 * 60
    time.sleep(wait)
def wait_x_hr(x=1):
    wait = x * 60 * 60
    time.sleep(wait)
    
def get_time_now():
    today = datetime.datetime.now()
    return today

def print_time_now():
    today = get_today()
    print(today)
    return today

def do_at_time(action, sec=None,min=None,hour=None, day=None, month=None, year=None):
	today = datetime.datetime.now()

	sleep = (datetime.datetime(today.year, today.month, today.day, 15, 20, 0) - today).seconds
	# print('Waiting for ' + str(datetime.timedelta(seconds=sleep)))
	time.sleep(sleep)
	return action

def wait_until(action, sec=None,min=None,hour=None, day=None, month=None, year=None):
	today = datetime.datetime.now()

	sleep = (datetime.datetime(today.year, today.month, today.day, 15, 20, 0) - today).seconds
	# print('Waiting for ' + str(datetime.timedelta(seconds=sleep)))
	time.sleep(sleep)
	return action

def wait_for(sec=0,min=0,hour=0):
	sleep = int(sec + min*60 + hour*60*60)
	time.sleep(sleep)

