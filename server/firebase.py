import time
from firebase_admin import credentials, firestore, initialize_app

cred = credentials.Certificate('key.json')
default_app = initialize_app(cred)
db = firestore.client()
data_ref = db.collection('data')


def save(data):
    try:
        data_ref.document(str(time.time())).set(data)
        return data
    except Exception as e:
        print(e)
