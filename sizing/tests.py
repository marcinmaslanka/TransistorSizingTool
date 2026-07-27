from django.test import TestCase
import numpy as np
import pandas as pd

from sizing.models import LUT
from sizing.models import CommonSource
from sizing.models import DesignSpecification
from sizing.models import Calculator

from unittest.mock import patch
import cProfile
import pstats

from sizing.models import DEFAULT_VDS

# Create your tests here.

class TestLUT(TestCase):

    def test_ut01_load_lut(self):

        """
        UnitTest-01: Verification of LUT file loading.

        The LUT class reads the file nmos.txt and stores the
        simulation data in the DataFrame NCH.

        This test verifies that the imported DataFrame contains
        all required columns in the expected order.
        """

        lut = LUT("data/nmos.txt")
        expected_columns = ["Length1", "cdd", "cgd", "cgg", "cgs", "gds", "gm", "Id", "Length", "Vth", "Vds", "Vgs"]

        self.assertFalse(lut.NCH.empty)
        self.assertListEqual(list(lut.NCH.columns), expected_columns)


class TestCommonSource(TestCase):

    """
    This test class verifies the correctness of the Common Source
    implementation based on the gm/Id design approach.

    The tests are divided into three levels:

    Unit Tests
    ----------
    - Verification of interpolation methods (VEA, gm/gds, Id/W)
    - Verification of analytical equations (Av, gm)
    - Verification of capacitance calculations

    Integration Tests
    -----------------
    - Verification of transistor width calculation
    - Verification of bandwidth calculation
    - Verification of interactions between LUT, Calculator and
      CommonSource classes

    System Tests
    ------------
    - Verification of complete sizing workflow
    - Comparison of calculated design results with NGSPICE
      simulation results
    - Verification of parameter sweep execution

    Test Environment
    ----------------
    Design Specification:
        VDD      = 1.21 V
        CL       = 5 pF
        RS       = 10 kΩ
        gm/Id    = 15 S/A
        L        = 130 nm
        RD       = 1 kΩ

    Technology Data:
        LUT file: data/nmos.txt
    """

    def setUp(self):

        self.spec = DesignSpecification(
            vdd=1.21,
            CL=5e-12,
            RS=10e3,
            gm_id_spec=15,
            L_spec=0.13e-6,
            RD_spec=1e3,
        )

        self.lut = LUT("data/nmos.txt")
        self.calculator = Calculator(lut=self.lut, width=5e-6)
        self.results = self.calculator.calculate()
        self.cs = CommonSource(self.spec, self.results)

    def test_ut02_calculate_vea(self):

        """
        UnitTest-02: Verification of VEA interpolation.

        The method calculate_vea() performs linear interpolation
        of id_gds_lut as a function of gm_id_lut. 
        
        For comparison the linear interpolation is done by hand below. 
        The data points are read from the graph.

        Calculation
        -----------

        gm_id_lut       id_gds_lut      # available Data Points
        13.74           1.23
        22.14           0.83

        vea = 1.23 + (15-13.74)/(22.14-13.74)*(0.83-1.23)       # formula for linear interpolation
        vea = 1.17
        """

        subset = pd.DataFrame({
            "gm_id_lut": [22.14, 13.74],
            "id_gds_lut": [0.83, 1.23]
        })

        vea = self.cs.calculate_vea(gm_id=self.spec.gm_id_spec, subset=subset)
        expected_vea = 1.17
        self.assertAlmostEqual(vea, expected_vea, places=2)

    def test_ut03_calculate_avmax(self):

        """
        UnitTest-03: Verification of maximum intrinsic gain selection.

        The method calculate_avmax() evaluates the intrinsic voltage gain
        for all available Vds operating points and returns:
            - the maximum achievable intrinsic gain Av0,
            - the corresponding Vds value.

        This test uses an artificial LUT dataset to verify the algorithmic
        behavior independently from real data.

        Test data:
            - A single transistor length is used because calculate_avmax()
            evaluates only one channel length.
            - Multiple Vds values are provided to verify the grouping and
            comparison.
            - Two gm/Id points per Vds value are provided to enable linear
            interpolation of VEA.

        Expected result are maximum intrinsic Gain Av0 and the corresponding Vds.
        """

        results = pd.DataFrame({
        "Length":       [0.13e-6,   0.13e-6,    0.13e-6,    0.13e-6 ],
        "Vds":          [0.2,       0.2,        0.6,        0.6     ],
        "gm_id_lut":    [10,        20,         10,         20      ],
        "id_gds_lut":   [2.0,       1.0,        1.0,        0.5     ]
        })

        cs = CommonSource(self.spec, results)
        av_table, av_max, best_vds = cs.calculate_avmax(15)
        #print(av_table)
        #print(av_max)
        #print(best_vds)

        self.assertEqual(len(av_table), 2)
        self.assertAlmostEqual(best_vds, 0.2, places=2)
        self.assertAlmostEqual(av_max, 9.05, places=2)

    def test_ut04_calculate_gmgds(self):

        """
        UnitTest-04: Verification of intrinsic gain interpolation.

        The method calculate_gm_gds() performs linear interpolation
        of gm_gds_lut as a function of gm_id_lut. The Value of gmgds Obtained from Graph.

        A synthetic dataset is used. The test verifies that the interpolation result is independent
        of the data ordering.
        """

        self.cs.results = pd.DataFrame({
        "Length":       [self.spec.L_spec,  self.spec.L_spec    ],
        "Vds":          [DEFAULT_VDS,       DEFAULT_VDS         ],
        "gm_id_lut":    [20,                12                  ],
        "gm_gds_lut":   [7,                 18                  ]
        })

        gm_gds = self.cs.calculate_gm_gds(gm_id=self.spec.gm_id_spec, vds=DEFAULT_VDS)

        # ficticious data points:
        #
        # gm/Id      gm/gds
        # 12          18
        # 20          7

        expected_gmgds = 13.875

        self.assertAlmostEqual(gm_gds, expected_gmgds, places=2)

    def test_ut05_calculate_gm(self):

        """
        UnitTest-05: Verification of transconductance calculation.

        The method calculate_gm() computes the required transistor
        transconductance from the maximum intrinsic gain Avmax,
        the intrinsic gain gm/gds and the load resistance RD.

        This test forces av and gmgds for purpose of calculating gm and compare
        the result with the Formula below.
        """

        av = 6.02
        gmgds = 17.13

        with patch.object(CommonSource, "calculate_gm_gds", return_value=gmgds):
            gm = self.cs.calculate_gm(self.spec.gm_id_spec, DEFAULT_VDS, av=av)

        expected_gm = ( (1/self.spec.RD_spec) / ((1/av) - (1/gmgds)) )

        self.assertAlmostEqual(gm, expected_gm, places=6)

    def test_ut06_calculate_idw(self):

        """
        UT-06: Verification of Id/W interpolation.

        The method calculate_id_w() performs linear interpolation
        of id_w_lut as a function of gm_id_lut.

        A synthetic dataset with known interpolation points is used
        to verify the interpolation algorithm.
        """

        self.cs.results = pd.DataFrame({
        "Length":       [self.spec.L_spec,      self.spec.L_spec    ],
        "Vds":          [DEFAULT_VDS,           DEFAULT_VDS         ],
        "gm_id_lut":    [12,                    20                  ],
        "id_w_lut":     [4,                     10                  ]
        })

        idw = self.cs.calculate_id_w(gm_id=15, vds=DEFAULT_VDS)

        # ficticious data points:
        #
        # gm/Id    Id/W
        # 12        4
        # 20       10

        expected_idw = 6.25

        self.assertAlmostEqual(idw, expected_idw, places=2)

    def test_ut07_calculate_cap(self):

        """
        UnitTest-07: Verification of parasitic capacitance interpolation.

        The methods calculate_cgs_w(), calculate_cgd_w() and
        calculate_cdd_w() perform linear interpolation of capacitance
        per width as a function of gm/Id.

        A synthetic dataset with known values is used to verify
        the interpolation.
        """

        self.cs.results = pd.DataFrame({
        "Length":       [self.spec.L_spec,  self.spec.L_spec    ],
        "Vds":          [DEFAULT_VDS,       DEFAULT_VDS         ],
        "gm_id_lut":    [10,                20                  ],
        "cgs_w_lut":    [100e-12,           200e-12             ],
        "cgd_w_lut":    [10e-12,            20e-12              ],
        "cdd_w_lut":    [20e-12,            40e-12              ]
        })

        cgsw = self.cs.calculate_cgs_w(gm_id=15, vds=DEFAULT_VDS)
        cgdw = self.cs.calculate_cgd_w(gm_id=15, vds=DEFAULT_VDS)
        cddw = self.cs.calculate_cdd_w(gm_id=15, vds=DEFAULT_VDS)

        self.assertAlmostEqual(cgsw, 150e-12)
        self.assertAlmostEqual(cgdw, 15e-12)
        self.assertAlmostEqual(cddw, 30e-12)

    def test_it01_calculate_W(self):

        """
        IntegrationTest-01: Verification of transistor width calculation.

        The complete sizing flow from intrinsic gain calculation
        to transistor width estimation is verified.

        The following components interact:
        - LUT
        - Calculator
        - CommonSource.calculate_avmax()
        - CommonSource.calculate_gm()
        - CommonSource.calculate_Id()
        - CommonSource.calculate_id_w()
        - CommonSource.calculate_W()

        The calculated transistor width is compared with a reference equation:

        W = Id / (Id/W)
        """

        _, av_max, best_vds = self.cs.calculate_avmax(self.spec.gm_id_spec)
        W = self.cs.calculate_W(gm_id=self.spec.gm_id_spec, vds=best_vds, av_max=av_max)

        gm = self.cs.calculate_gm(self.spec.gm_id_spec, best_vds, av_max)
        Id = gm / self.spec.gm_id_spec
        Id_W = self.cs.calculate_id_w(self.spec.gm_id_spec, best_vds)
        expected_W = Id / Id_W

        self.assertAlmostEqual(W, expected_W, delta=expected_W * 0.01)
        
    def test_it02_calculate_bw(self):

        """
        IntegrationTest-02: Verification of bandwidth calculation.

        This test verifies the complete bandwidth calculation chain:
        LUT -> interpolation -> transistor sizing ->
        parasitic capacitance estimation ->
        output resistance calculation ->
        bandwidth calculation.

        The calculated bandwidth is compared against a reference
        value obtained from analytical calculation.
        """

        av = 6.02

        bw = self.cs.calculate_bw(gm_id=self.spec.gm_id_spec, vds=DEFAULT_VDS, av_max=av)

        W = self.cs.calculate_W(gm_id=self.spec.gm_id_spec, vds=DEFAULT_VDS, av_max=av)
        cgs = W * self.cs.calculate_cgs_w(self.spec.gm_id_spec, DEFAULT_VDS)
        cgd = W * self.cs.calculate_cgd_w(self.spec.gm_id_spec, DEFAULT_VDS)
        cdd = W * self.cs.calculate_cdd_w(self.spec.gm_id_spec, DEFAULT_VDS)
        cdb = cdd - cgd
        Rout = self.cs.calculate_Rout(self.spec.gm_id_spec, DEFAULT_VDS, av)
        tau = (self.spec.RS *(cgs + cgd*(1+av)) + Rout * (self.spec.CL+cdb+cgd) )
        expected_bw = 1/(2*np.pi*tau)

        self.assertAlmostEqual(bw, expected_bw, delta=expected_bw*0.01)

    def test_st01_target_spec(self):

        """
        SystemTest-01: Validation of the complete amplifier design flow.

        The transistor sizing process is executed for a defined design
        specification. The calculated amplifier parameters are compared
        with reference values obtained from ngspice simulation.

        Tested design point:
            gm/Id = 15
            RD = 1 kOhm
            L = 130 nm
            VDD = 1.21 V
            CL = 5 pF

        Verified system parameters:
            - Intrinsic voltage gain Av0
            - -3dB bandwidth BW
        """

        results = self.cs.size_sweep(gmid_min=15, gmid_max=15, RD_min=1000, RD_max=1000)
        self.assertEqual(len(results), 1)
        row = results.iloc[0]
        ngspice_av0 = 6.63
        ngspice_bw = 58.34e6

        self.assertAlmostEqual(row["av0"], ngspice_av0, delta=ngspice_av0*0.02)     # 2% tolerance 
        self.assertAlmostEqual(row["bw"], ngspice_bw, delta=ngspice_bw*0.02)

class TestViews(TestCase):

    def test_st01_post_request(self):

        """
        SystemTest-01: Verification of the complete web-based design flow.

        This test simulates a user submitting a valid amplifier design
        specification through the Django web interface.

        The test verifies the complete interaction between:
            - HTTP POST request
            - Formular validation
            - DesignSpecification creation
            - CommonSource sizing algorithm
            - Plot generation
            - Template context generation

        Test input:
            VDD      = 1.21 V
            CL       = 5 pF
            RS       = 10 kOhm
            gm/Id    = 15
            L        = 130 nm
            RD       = 1 kOhm

        Expected result:
            - HTTP response status is 200
            - Submitted form is valid
            - Bandwidth plot is generated
            - Maximum gain plot is generated
        """

        response = self.client.post(
            "/",
            {
                "vdd": 1.21,
                "CL": 5e-12,
                "RS": 10000,
                "gm_id_spec": 15,
                "L_spec": 0.13e-6,
                "RD_spec": 1000,
            }
        )

        self.assertEqual(response.status_code, 200)

        self.assertTrue(response.context["formular"].is_valid())

        self.assertIsNotNone(response.context["graph1"])
        self.assertIsNotNone(response.context["graph7"])

class TestPerformance(TestCase):

    def test_profile_size_sweep(self):

        self.spec = DesignSpecification(
                    vdd=1.21,
                    CL=5e-12,
                    RS=10e3,
                    gm_id_spec=15,
                    L_spec=0.13e-6,
                    RD_spec=1e3,
                )

        self.lut = LUT("data/nmos.txt")
        self.calculator = Calculator(lut=self.lut, width=5e-6)
        self.results = self.calculator.calculate()
        self.cs = CommonSource(self.spec, self.results)

        cProfile.run("cs.calculate_avmax(15)")


    
