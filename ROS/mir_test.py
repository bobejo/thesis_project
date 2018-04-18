import requests
import json
import time

"""
A program to run MIR to a point and return it back to the starting point
"""

"""
To get the ID of the missions
"""
# Get mission ID
# resp = requests.get('http://192.168.1.140:8080/v1.0.0/missions')
# print resp.json()

# {u'url': u'/v1.0.0/missions/942d5de5-398f-11e8-9162-f44d306bb564', u'guid': u'942d5de5-398f-11e8-9162-f44d306bb564', u'name': u'MoveToPlacing'}
# {u'url': u'/v1.0.0/missions/a726b09f-398f-11e8-9162-f44d306bb564', u'guid': u'a726b09f-398f-11e8-9162-f44d306bb564', u'name': u'MovetoINitial'}



class Mir:
	def __init__(self):
		self.charging=0


	def reset_PLC_registers(self):
		taska = {"value" : "0"}
		respa = requests.post('http://192.168.1.140:8080/v1.0.0/registers/1', json=taska)
		taskb = {"value" : "0"}
		respb = requests.post('http://192.168.1.140:8080/v1.0.0/registers/2', json=taskb)
		taskc = {"value" : "0"}
		respc = requests.post('http://192.168.1.140:8080/v1.0.0/registers/3', json=taskc)
		taskd = {"value" : "0"}
		respd = requests.post('http://192.168.1.140:8080/v1.0.0/registers/4', json=taskd)
		taske = {"value" : "0"}
		respe = requests.post('http://192.168.1.140:8080/v1.0.0/registers/5', json=taske)
		taskf = {"value" : "0"}
		respf = requests.post('http://192.168.1.140:8080/v1.0.0/registers/6', json=taskf)


	# Set desired MiR state
	def mir_executing_state(self):
		task = {"state" : 5}
		resp = requests.put('http://192.168.1.140:8080/v1.0.0/state', json=task)


	def mir_start_movement(self):
		task = {"value" : "1"}
		resp = requests.post('http://192.168.1.140:8080/v1.0.0/registers/6', json=task)

	# Set desired MiR state
	def mir_pause_state(self):
		task = {"state" : 4}
		resp = requests.put('http://192.168.1.140:8080/v1.0.0/state', json=task)

	# Mission go to Placing Point
	def mission_1_to_queue(self):
		task ={"mission" : "942d5de5-398f-11e8-9162-f44d306bb564"}
		resp = requests.post('http://192.168.1.140:8080/v1.0.0/mission_queue', json=task)
		# self.charging=0



	# Mission go to initial Point
	def mission_2_to_queue(self):
		task ={"mission" : "a726b09f-398f-11e8-9162-f44d306bb564"}
		resp = requests.post('http://192.168.1.140:8080/v1.0.0/mission_queue', json=task)
		# self.charging=0


	# Delete the Mission Queue
	def delete_mir_mission_queue(self):
		resp = requests.delete('http://192.168.1.140:8080/v1.0.0/mission_queue')


	def demand_1_tube(self):
		task = {"value" : "1"}
		resp = requests.post('http://192.168.1.140:8080/v1.0.0/registers/1', json=task)



	# Move to Placing Position to get objects
	def start_mission_1(self):
		self.delete_mir_mission_queue()
		time.sleep(1)

		self.reset_PLC_registers()
		time.sleep(1)

		self.mission_1_to_queue()
		time.sleep(1)
		self.mir_executing_state()

	# Move to Initial point
	def start_mission_2(self):
		self.delete_mir_mission_queue()
		time.sleep(1)

		self.reset_PLC_registers()
		time.sleep(1)

		self.mission_2_to_queue()
		time.sleep(1)
		self.mir_executing_state()



# a=Mir()
# a.start_mission_2()
