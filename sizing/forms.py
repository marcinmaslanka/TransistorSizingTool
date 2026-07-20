from django import forms


class Formular(forms.Form):

    vdd = forms.FloatField(label="Supply Voltage (V)",
                           initial=1.2,
                           disabled=True,
                           )

    CL = forms.FloatField(label="CL (F)",
                          initial=5e-12,
                          min_value=1e-12,
                          max_value=500e-12,
                          error_messages={
                              "min_value" : "CL can not be less than 1 pF",
                              "max_value" : "CL can not be greather than 500pF",
                          })
    
    RS = forms.FloatField(label="RS (Ω)", 
                          initial=10e3,
                          disabled=True,
                          )

    gm_id_spec = forms.FloatField(label="gm/Id", 
                                  initial=15,
                                  min_value=5,
                                  max_value=25,
                                  error_messages={
                                      "min_value" : "gm/Id can not be less than 5",
                                      "max_value" : "gm/Id can not be greather than 25",
                                })

    L_spec = forms.ChoiceField(label="Channel Length")

    def __init__(self, *args, lengths=None, **kwargs):
        
        super().__init__(*args, **kwargs)

        if lengths is not None:
            self.fields["L_spec"].choices = [
                (L, f"{L*1e6:.2f}")
                for L in lengths
            ]

    RD_spec = forms.FloatField(label="RD (Ω)", 
                               initial=1e3,
                               min_value=0.5e3,
                               max_value=12e3,
                               error_messages={
                                   "min_value" : "RD can not be less than 500Ohm",
                                   "max_value" : "RD can not be greather than 12kOhm",
                                })
    
