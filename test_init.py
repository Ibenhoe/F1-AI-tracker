import requests
import json

try:
    r = requests.get('http://localhost:5000/api/race/init?race=21', timeout=10)
    print('Status:', r.status_code)
    
    if r.status_code == 200:
        d = r.json()
        print('Race:', d.get('race_name'))
        drivers = d.get('drivers', [])
        print('Total drivers:', len(drivers))
        
        if len(drivers) > 0:
            print('\nFirst 3 drivers:')
            for drv in drivers[:3]:
                print(f"  {drv.get('code')} - {drv.get('name')} ({drv.get('team')})")
            print('...')
            print(f'\nLast driver: {drivers[-1].get("code")} - {drivers[-1].get("name")}')
    else:
        print('Error:', r.text[:500])
except Exception as e:
    print('ERROR:', str(e))
