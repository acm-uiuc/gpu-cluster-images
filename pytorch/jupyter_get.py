from notebook import notebookapp
import time

count = 0
is_failed = False

time.sleep(1)
notebooks = list(notebookapp.list_running_servers())

while len(notebooks) == 0:
    count += 1
    #print(count)
    #if (count == 500):
    #    is_failed = True
    #    break
    notebooks = list(notebookapp.list_running_servers())
    pass

if (is_failed is not True):
    note = notebooks[0]
    print(note.get('token', ''))
