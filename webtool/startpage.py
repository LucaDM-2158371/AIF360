from flask import Flask, render_template  
app = Flask(__name__)   # Flask constructor 
  
# A decorator used to tell the application 
# which URL is associated function
@app.route('/')       
def homepage(): 
    return render_template('templates/index.html')
  
if __name__=='__main__': 
   app.run() 
