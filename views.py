from django.shortcuts import render
#from .apps import PredictorConfig
#from django.http import JsonResponse
from django.views.generic import TemplateView,FormView
from django.contrib.staticfiles.storage import staticfiles_storage
from predictor.forms import SalaryPredictor
import pickle
import numpy as np

class Salary(TemplateView):
    template_name='home.html'

    def prediction(self,inputs,val):
        '''model_file = staticfiles_storage.path('models.pkl')
        with open(model_file, 'rb') as fid:
            pickl = pickle.load(fid)
        predictions = pickl.predict(inputs)
        return predictions'''
        if val=='a1':
            m1=staticfiles_storage.path('model1.pkl')
            with open(m1,'rb') as f1:
                p1=pickle.load(f1)
            predictions1=p1.predict(inputs)
            return predictions1
        elif val=='a2':
            m2=staticfiles_storage.path('model2.pkl')
            with open(m2,'rb') as f2:
                p2=pickle.load(f2)
            predictions2=p2.predict(inputs)
            return predictions2
        elif val=='a3':
            m3=staticfiles_storage.path('model3.pkl')
            with open(m3,'rb') as f3:
                p3=pickle.load(f3)
            predictions3=p3.predict(inputs)
            return predictions3



    def get(self,request):
        form=SalaryPredictor()
        return render(request,self.template_name, {'form':form})

    def post(self,request):
        #if request.method=='POST':
        form=SalaryPredictor(request.POST)
        if form.is_valid():
                #crim = form.cleaned_data['CRIM']
            variables = [i for i in form.cleaned_data.values()]
            #variables = np.array(variables).astype('float').reshape(1,-1)
            pred1 = self.prediction([variables],'a1')
            pred1=np.array([int(pred1)]).reshape(1,1)
            pred1 = pred1.item()

            pred2 = self.prediction([variables],'a2')
            pred2=np.array([int(pred2)]).reshape(1,1)
            pred2 = pred2.item()

            pred3 = self.prediction([variables],'a3')
            pred3=np.array([int(pred3)]).reshape(1,1)
            pred3 = pred3.item()


            args = {'form':form,'pred1':pred1,'pred2':pred2,'pred3':pred3}
            return render(request,self.template_name,args)
