from flask import Flask

# Inisialisasi objek Flask
app = Flask(__name__, template_folder="templates")

# Impor modul routes untuk menambahkan rute ke aplikasi Flask
from app import routes
