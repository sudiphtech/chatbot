import urllib.request as u
r = u.urlopen('http://127.0.0.1:8000/analytics/import-students/')
print(r.status)
print(r.read().decode())
