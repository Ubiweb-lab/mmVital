from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

import pandas as pd
import os
import numpy as np
from core.signalProcess.phasesignal_vitalsign import *
from types import SimpleNamespace
from django.views import View


@method_decorator(csrf_exempt, name='dispatch')  # Decorator to exempt CSRF for this view
class RadarView(View):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_directory = os.path.dirname(os.path.abspath(__file__))
        self.const = SimpleNamespace(DATA_SOURCE = 0) # 0: hex data,   1: signal process
        self.is_reload = True 

        self.HR_data_pool = np.array([])
        self.RR_data_pool = np.array([])


    def initial_data_pool(self, request, index_id=0):
        
        # print(len(request.session.get('HR_data_pool', [])))
        pool_size = len(request.session.get('HR_data_pool', []))
        print(f'pool size init = {pool_size}')

        if(pool_size == 0 or (self.is_reload is True and index_id==0)):
            request.session.clear()

            print(f'pool size init 2 = {len(request.session.get("HR_data_pool", []))}')

            if(self.const.DATA_SOURCE == 0):
                HR_path = os.path.join(self.root_directory, "static/waves/heart_rate.csv")
                RR_path = os.path.join(self.root_directory, "static/waves/breathing_rate.csv")

                self.HR_data_pool = pd.read_csv(HR_path).to_numpy()[:, 1]
                self.RR_data_pool = pd.read_csv(RR_path).to_numpy()[:, 1]

                request.session['HR_data_pool'] = self.HR_data_pool.tolist()
                request.session['RR_data_pool'] = self.RR_data_pool.tolist()

            elif(self.const.DATA_SOURCE == 1):
                # self.HR_data_pool, self.RR_data_pool = signal_process()
                HR_RR_path = os.path.join(self.root_directory, "static/waves/HR_RR.csv")
                HR_RR_data = pd.read_csv(HR_RR_path).to_numpy()
                self.HR_data_pool = HR_RR_data[:, 0]
                self.RR_data_pool = HR_RR_data[:, 1]

                request.session['HR_data_pool'] = self.HR_data_pool.tolist()
                request.session['RR_data_pool'] = self.RR_data_pool.tolist()
        else:
            self.HR_data_pool = np.array(request.session.get('HR_data_pool', []))
            self.RR_data_pool = np.array(request.session.get('RR_data_pool', []))


    ##  Monitor view from here
    def get(self, request, *args, **kwargs):
        return render(request, "monitor.html", {"status": 1})

   
    def post(self, request, *args, **kwargs):
        parms = request.POST
        index_id = int(parms.get("index_id")) 
        print(f"index_id = {index_id}")
        self.initial_data_pool(request, index_id = index_id)
        HR = self.HR_data_pool[index_id]
        RR = self.RR_data_pool[index_id]
        
        return JsonResponse({"HR": HR, "RR": RR})
