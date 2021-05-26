from tkinter import *
from PIL import ImageTk,Image
import sqlite3
from tkinter import messagebox as ms





with sqlite3.connect('quit.db') as dbs:
    cc = dbs.cursor()

cc.execute('CREATE TABLE IF NOT EXISTS user (username TEXT NOT NULL PRIMARY KEY,password TEX NOT NULL);')
dbs.commit()
dbs.close()
im=Image.open("clas.png")

class main:
    def __init__(self,master):
        self.master = master
        
        self.username = StringVar()
        self.password = StringVar()
        self.n_username = StringVar()
        self.n_password = StringVar()
      
        self.widgets()

    #Login Function
    def login(self):
    	
        with sqlite3.connect('quit.db') as db:
            c = db.cursor()

        if len(self.username.get())!=0 and len(self.password.get())!=0 and self.username.get()!=" ":
            F_user = ('SELECT * FROM user WHERE username = ? and password = ?')
            c.execute(F_user,[(self.username.get()),(self.password.get())])
            result = c.fetchall()
            if result:
                root.destroy()
                import mt
                #m.call()
                #m.tryt()
            else:
                ms.showerror('Oops!','Username Not Found.')

        else:
            ms.showwarning("Error","Enter Valid username or Password")

  

            
    def new_user(self):
    	
        with sqlite3.connect('quit.db') as db:
            c = db.cursor()

        if len(self.n_username.get())!=0 and len(self.n_password.get())!=0 and self.n_username.get()!=" ":
            F_user = ('SELECT username FROM user WHERE username = ?')
            c.execute(F_user,[(self.n_username.get())])        
            if c.fetchall():
                ms.showerror('Error!','Username is Taken ')
            else:
                ms.showinfo('Success!','Account Created Successfully!')
                self.log()
        #Create New Account 
            insert = 'INSERT INTO user(username,password) VALUES(?,?)'
            c.execute(insert,[(self.n_username.get()),(self.n_password.get())])
            db.commit()
        else:
            ms.showwarning("Error","Enter Valid username or Password")

     


        
    def log(self):
        #login data
        self.username.set('')
        #password
        self.password.set('')
        self.new.pack_forget()
        self.head['text'] = 'LOGIN'
        self.flog.pack()
    def cr(self):
        #creatipon of login
        self.n_username.set('')
        #pasword at creation time
        self.n_password.set('')
        self.flog.pack_forget()
        self.head['text'] = 'Create Account'
        self.new.pack()
        
    #Draw Widgets
    def widgets(self):
        
        self.head = Label(self.master,text = 'ATTANDANCE SYSTEM',font = ('Lucida Calligraphy',35),pady = 10)
        self.head.pack()
        self.head = Label(self.master,text = 'LOGIN',font = ('Book Antiqua',25),pady = 10)
        self.head.pack()
        self.flog = Frame(self.master,padx =10,pady = 10)
        Label(self.flog,text = 'Username: ',font = ('Book Antiqua',20),pady=5,padx=5).grid(row=0,column=1)
        Entry(self.flog,textvariable = self.username,bd = 5,font = ('Book Antiqua',15)).grid(row=0,column=2)
        Label(self.flog,text = 'Password: ',font = ('Book Antiqua',20),pady=5,padx=5).grid(row=2,column=1)
        Entry(self.flog,textvariable = self.password,bd = 5,font = ('Book Antiqua',15),show = '*').grid(row=2,column=2)
        Button(self.flog,text = ' Login ',bd = 3 ,font = ('Book Antiqua',15),padx=5,pady=5,command=self.login).grid(row=4,column=1)
        Button(self.flog,text = ' Create Account ',bd = 3 ,font = ('Book Antiqua',15),padx=5,pady=5,command=self.cr).grid(row=4,column=2)
        self.flog.pack()
        
        self.new = Frame(self.master,padx =10,pady = 10)
        Label(self.new,text = 'Username: ',font = ('Book Antiqua',20),pady=5,padx=5).grid(sticky = W)
        Entry(self.new,textvariable = self.n_username,bd = 5,font = ('Book Antiqua',15)).grid(row=0,column=1)
        Label(self.new,text = 'Password: ',font = ('Book Antiqua',20),pady=5,padx=5).grid(sticky = W)
        Entry(self.new,textvariable = self.n_password,bd = 5,font = ('Book Antiqua',15),show = '*').grid(row=1,column=1)
        Button(self.new,text = 'Create Account',bd = 3 ,font = ('Book Antiqua',15),padx=5,pady=5,command=self.new_user).grid()
        Button(self.new,text = 'Return to Login',bd = 3 ,font = ('Book Antiqua',15),padx=5,pady=5,command=self.log).grid(row=2,column=1)
if __name__ == '__main__':
    
    root = Tk()
    root.title('Login Form')
    root.geometry('900x700')

    canvas = Canvas(root ,width = 900, height = 380)
    canvas.pack(side=BOTTOM)
    
    im=im.resize((900,380))
    im=ImageTk.PhotoImage(im)
    canvas.create_image(0, 0, anchor=NW, image=im)
   # Label(root,text = 'IMAGE',image=im).pack(side=BOTTOM)
    main(root)
    root.mainloop()