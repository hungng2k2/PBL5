import time
from firebase_admin import credentials, firestore, initialize_app

DURATION_IGNORE_SAME_LABEL_SECONDS = 60

cred = credentials.Certificate('key.json')
default_app = initialize_app(cred)
db = firestore.client()
data_ref = db.collection('data')


def save(info):
    try:
        if len(info['labels']) > 0:
            info_map =  {
                "hostname": info['hostname'],
                "timeseconds": info['timeseconds'],
                "time": info['time'],
                "labels": {label: info['labels'].count(label) for label in set(info['labels'])}
            }
            # Get last data has same labels with current data
            query = data_ref.where('labels', u'==', info_map['labels']).order_by(u'timeseconds', direction=firestore.Query.DESCENDING).limit(1)
            result = query.get()
            # If result is empty, upload current data
            if len(result) == 0:
                data_ref.document(str(time.time())).set(info_map)
                return info
            last_data_with_same_labels = result[0].to_dict()
            # If last upload time exceeds DURATION_IGNORE_SAME_LABEL_SECONDS upload current data
            if (info_map['timeseconds'] - last_data_with_same_labels['timeseconds'] > DURATION_IGNORE_SAME_LABEL_SECONDS):
                data_ref.document(str(time.time())).set(info_map)
                return info
    except Exception as e:
        print(e)