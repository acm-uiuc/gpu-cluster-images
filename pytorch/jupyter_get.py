from notebook import notebookapp

note = next(notebookapp.list_running_servers())

print(note.get('token', ''))
