
from django.http import HttpResponse
from django.shortcuts import render
from .models import LUT, Plot, Equations, Calculator, DesignSpecification, CommonSource, lut, RESULTS, Vectors
from .models import GRAPH2, GRAPH3, GRAPH4, GRAPH5
from .forms import Formular

def index(request):         # entgegenahme der Benutzer anfrage nach dem aufrufen der WEbseite

    formular = Formular()   #erzeugen eines leeren formulars
    graph1 = None
    graph2 = None
    graph3 = None
    graph4 = None
    graph5 = None
    graph6 = None
    graph7 = None

    if request.method == "POST":

        formular = Formular(request.POST, lengths=lut.get_lengths()) # einleses des Formulars

        if formular.is_valid():             
            spec = DesignSpecification(             #übergabe des Formulars an das Objekt spec
                vdd=formular.cleaned_data["vdd"],
                CL=formular.cleaned_data["CL"],
                RS = formular.cleaned_data["RS"],
                gm_id_spec = formular.cleaned_data["gm_id_spec"],
                L_spec = float(formular.cleaned_data["L_spec"]),
                RD_spec = float(formular.cleaned_data["RD_spec"]),
            )
        
            cs = CommonSource(spec, RESULTS)
            rd= spec.RD_spec
            df= cs.size_sweep(gmid_min=5, gmid_max=25, gmid_step=5, RD_min=rd, RD_max=rd, RD_step=rd)

            #print(df.to_string())
            df_avmax, _, _ = cs.calculate_avmax(spec.gm_id_spec)
            
            #diagrame
            plotter = Plot()
            graph1 = plotter.plot_bw_id(df)
            graph2 = GRAPH2
            graph3 = GRAPH3
            graph4 = GRAPH4
            graph5 = GRAPH5
            graph7 = plotter.plot_avmax_vds(df_avmax)
            print(type(graph1))
            


    return render(                  #übergabe an das Template
        request,
        "sizing/index.html",
        {
            "graph1": graph1,
            "graph2": graph2,
            "graph3": graph3,
            "graph4": graph4,
            "graph5": graph5,
            "graph7": graph7,
            "formular": formular,
        }
    )