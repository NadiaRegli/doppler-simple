#include "dopplerObjectiveFunction.h"

DopplerObjectiveFunction::DopplerObjectiveFunction(double sigma, Eigen::MatrixXd scan, std::vector<double> scanVel, Eigen::Matrix4d Tcalib) 
{
    sigma_ = sigma;
    scan_ = scan;
    scanSize_ = scan.rows();
    scanVel_ = scanVel;
    Tcalib_ = Tcalib;
}

double DopplerObjectiveFunction::operator()(const column_vector& m) const
{

    // Load the transform from vehicle to sensor
    Eigen::Matrix4d T_V_to_S = Tcalib_;

    // Period of LiDAR sampling
    double period = 0.1;

    // transformation hypothesis from the new scan to the local map
    Eigen::Matrix4d hypothesis = homogeneous(m(0), m(1), m(2), m(3), m(4), m(5));
    
    // set the score to zero
    double score = 0;
    std::vector<double> scores(scanVel_.size(), 0.0);
    size_t numResults = 1;
    // calculate the hypothesis reward 
    // loop through each transformed target scan and find the closest point in the source scan
    for (unsigned int i = 0; i < scanVel_.size(); i++)
    {   
        const Eigen::VectorXd state_vector = hom2rpyxyz_v2(hypothesis);
        // std::cout << state_vector << std::endl;
        //Extrinsic calibration --> RVL 
        const Eigen::Matrix3d R_S_to_V = T_V_to_S.block<3, 3>(0, 0).inverse();
        //Extrinsic calibration --> VtVL
        const Eigen::Vector3d r_v_to_s_in_V = T_V_to_S.block<3, 1>(0, 3);
        //VwIV = -u(theta)/delta t    (eq16)
        const Eigen::Vector3d w_v_in_V = state_vector.block<3, 1>(0, 0) / period;
        //VvV = -u(t)/delta t    (eq17)
        const Eigen::Vector3d v_v_in_V = state_vector.block<3, 1>(3, 0) / period;
        //VvL = VvV + VwIV cross VtVL    (eq7)
        const Eigen::Vector3d v_s_in_V = v_v_in_V + w_v_in_V.cross(r_v_to_s_in_V);
        // LvL = RVL*VvL     (eq13fixed)
        const Eigen::Vector3d v_s_in_S = R_S_to_V * v_s_in_V;

        // Vmeas from the scan file
        const double doppler_in_S = scanVel_[i];
        //Direction vector of LiDAR to Measured Point in FL (range measurement)
        const Eigen::Vector3d LtLP = {scan_.coeffRef(i,0), scan_.coeffRef(i,1), scan_.coeffRef(i,2)};
        //LdLP (direction vector)
        const Eigen::Vector3d ds_in_V = LtLP/LtLP.norm();

        // Compute predicted Doppler velocity.
        //VdLP = RVL dot LdLP  (eq14)
        const Eigen::Vector3d ds_in_S = R_S_to_V * ds_in_V;
        //LvLP = -VdLP dot RVL(vVv + VwIV cross VtVL) (eq15fixed)
        const double doppler_pred_in_S = -ds_in_S.dot(v_s_in_S);
        //rvj = vmeasj - LvLPj(u)    (eq18)
        const double doppler_error = abs(doppler_in_S - doppler_pred_in_S);
        scores[i] = exp(-pow(doppler_error, 2)/(2*pow(sigma_, 2)));
    }


    // deterministic results, sum the scores
    for (auto& n: scores)
        score -= n;
    return score;

}