from tkinter import ttk, filedialog
import tkinter as tk
import time
from datetime import date, datetime
from tkinter import messagebox as mb
from threading import Thread
import os, django 
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance.settings')
django.setup()
from core.recognition import *

def print_time():
    datetime_label.config(text="DATE: " + str(datetime.now().strftime("%d/%m/%Y")) + "        " +
                          "TIME: " + str(datetime.now().strftime("%H:%M:%S")))
    datetime_label.after(1000, print_time)

def attendance():
    # Thread(target=recognize).start()
    recognize()
    return

def save_face(emb):
    save_embedding(emb, str(name.get()))
    name_entry.unbind("<Return>")
    name_entry.delete(0, tk.END)
    name_entry.grid_remove()
    name_label.grid_remove()


def capture_face():
    q, emb = capture()
    if q:
        name_entry.focus()
        name_label.grid(row=4, column=0, columnspan=2)
        name_entry.grid(row=4, column=2, columnspan=4, padx=25, pady=4)
        name_entry.bind('<Return>', lambda event: save_face(emb))


def add_face_from_folder():
    folder_selected = filedialog.askdirectory()
    print(folder_selected)
    if folder_selected:
        save_from_folder(folder_selected)


def add_face_from_file():
    file_selected = filedialog.askopenfilename()
    print(file_selected)
    filename = str(file_selected.split("/")[-1])
    extension = filename.split(".")[-1]
    name = filename.split(".")[0]
    if file_selected and extension in ["jpeg", "png", "jpg"]:
        name = str(file_selected.split("/")[-1]).split(".")[0]
        print(name)
        save_from_file(file_selected, name)
    else:
        mb.showerror("ERROR", "FILE NOT AN IMAGE")


root = tk.Tk()
root.title("FACE RECOGNITION")
root.geometry(f"1050x300+100+100")
root.configure(background='#efeff1')
tk.Label(root, text="FACE RECOGNITION ATTENDANCE SYSTEM",
            fg="white", bg="#ff725e",
            font=("Helvetica", 24, "bold")
            ).grid(row=0, columnspan=8, padx=25, pady=20)

name = tk.StringVar(root)
datetime_label = tk.Label(root, text="DATE: " + str(datetime.now().strftime("%d/%m/%Y")) + "        " +
                            "TIME: " + str(time.strftime("%H:%M:%S")),
                            fg="black", bg='#efeff1',
                            font=("Helvetica", 12)
                            )
take_attendance_button = tk.Button(
    root, text="Take attendance", bg="#ff725e", fg="black", font=("Helvetica", 14, "bold"), command=attendance)
capture_face_button = tk.Button(
    root, text='Capture and add face', bg="#ff725e", fg="black", font=("Helvetica", 14, "bold"), command=capture_face)
add_new_face_file_button = tk.Button(
    root, text='Add face from image', bg="#ff725e", fg="black", font=("Helvetica", 14, "bold"), command=add_face_from_file)
add_new_face_folder_button = tk.Button(
    root, text='Add face(s) from folder', bg="#ff725e", fg="black", font=("Helvetica", 14, "bold"), command=add_face_from_folder)

name_label = tk.Label(root, text="Name of person: ",
                        bg='#efeff1', fg="black",
                        font=("Helvetica", 12)
                        )
name_entry = ttk.Entry(root, width=20, text='', textvariable=name)

datetime_label.grid(row=1, columnspan=8, padx=25, pady=4)
take_attendance_button.grid(row=2, column=0, columnspan=2, padx=25, pady=4)
add_new_face_file_button.grid(row=2, column=2, columnspan=2, padx=25, pady=4)
add_new_face_folder_button.grid(row=2, column=4, columnspan=2, padx=25, pady=4)
capture_face_button.grid(row=2, column=6, columnspan=2, padx=25, pady=4)
print_time()
att = at.Attendance()
att.schedule_write_attendance_to_db(root)
root.mainloop()



