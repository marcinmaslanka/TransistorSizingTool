
from django.http import HttpResponse
from django.shortcuts import render
from .models import Plot, DesignSpecification, CommonSource, lut, RESULTS
from .models import GRAPH2, GRAPH3, GRAPH4, GRAPH5, GRAPH6
from .forms import Formular

def index(request):

    """
    Main view of the Transistor Sizing Tool.

    Handles HTTP GET and POST requests. For GET requests, an input form
    with default design specifications is displayed. For POST requests,
    the submitted parameters are validated, a common-source amplifier is
    sized using the gm/Id methodology, and the resulting design-space
    data and plots are generated.

    Parameters
    ----------
    request : HttpRequest
        Incoming HTTP request containing optional form data.

    Returns
    -------
    HttpResponse
        Rendered HTML page containing the input form and generated plots.

    """

    formular = Formular(lengths=lut.get_lengths())
    graph1 = None
    graph2 = None
    graph3 = None
    graph4 = None
    graph5 = None
    graph6 = None
    graph7 = None

    if request.method == "POST":

        formular = Formular(request.POST, lengths=lut.get_lengths())

        if formular.is_valid():             
            spec = DesignSpecification(
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
            
            plotter = Plot()
            graph1 = plotter.plot_bw_id(df)
            graph2 = GRAPH2
            graph3 = GRAPH3
            graph4 = GRAPH4
            graph5 = GRAPH5
            graph6 = GRAPH6
            graph7 = plotter.plot_avmax_vds(df_avmax)

    return render(
        request,
        "sizing/index.html",
        {
            "graph1": graph1,
            "graph2": graph2,
            "graph3": graph3,
            "graph4": graph4,
            "graph5": graph5,
            "graph6": graph6,
            "graph7": graph7,
            "formular": formular,
        }
    )