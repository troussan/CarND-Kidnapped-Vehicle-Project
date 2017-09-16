/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define TWO_PI (2.0*M_PI)

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 100;
  default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    double sample_x, sample_y, sample_theta;

    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);

    Particle p;
    p.id = i;
    p.x = sample_x;
    p.y = sample_y;
    p.theta = sample_theta;
    p.weight = 1.0;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;

  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);

  for (auto &p: particles) {
    if (fabs(yaw_rate) > 0.0001) {
      double v_over_yr = velocity / yaw_rate;
      double theta = p.theta + yaw_rate * delta_t;
      p.x += v_over_yr * (sin(theta) - sin(p.theta));
      p.y += v_over_yr * (cos(p.theta) - cos(theta));
      p.theta = theta;
    } else {
      p.x += velocity * cos(p.theta) * delta_t;
      p.y += velocity * sin(p.theta) * delta_t;
    }

    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  std::vector<LandmarkObs> matched;
  for (auto &pr: predicted) {
    double min_dist = 1.0E+100;
    LandmarkObs *closest;
    for (auto &o: observations) {
      double d = dist(pr.x, pr.y, o.x, o.y);
      if (d < min_dist) {
        closest = &o;
        min_dist = d;
      }
    }
    if (closest != NULL) {
      closest->id = pr.id;
      matched.push_back(*closest);
    }
  }
  observations.clear();
  observations = matched;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  double weight_mult = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
  double sigma_2_x = 2.0 * std_landmark[0] * std_landmark[0];
  double sigma_2_y = 2.0 * std_landmark[1] * std_landmark[1];

  weights.clear();
  for (auto &p: particles) {
    std::vector<LandmarkObs> observations_map;

    double sin_theta = sin(p.theta);
    double cos_theta = cos(p.theta);

    for (auto &o: observations) {
      LandmarkObs obs;
      obs.x = p.x + cos_theta * o.x - sin_theta * o.y;
      obs.y = p.y + sin_theta * o.x + cos_theta * o.y;
      observations_map.push_back(obs);
    }

    std::vector<LandmarkObs> predicted;
    for (auto &l: map_landmarks.landmark_list) {
      if (dist(p.x, p.y, l.x_f, l.y_f) <= sensor_range) {
        LandmarkObs landmark;
        landmark.id = l.id_i;
        landmark.x = l.x_f;
        landmark.y = l.y_f;
        predicted.push_back(landmark);
      }
    }

    dataAssociation(predicted, observations_map);

    std::vector<int> assoc;
    std::vector<double> x_sense;
    std::vector<double> y_sense;
    double weight = 1.0;
    if (observations_map.size() > 0) {
      for (auto &o : observations_map) {
        bool found = false;
        assoc.push_back(o.id);
        x_sense.push_back(o.x);
        y_sense.push_back(o.y);
        for (auto &pr : predicted) {
          if (o.id == pr.id) {
            weight *= weight_mult * exp(-(pow(o.x - pr.x, 2) / sigma_2_x + pow(o.y - pr.y, 2) / sigma_2_y));
            found = true;
            break;
          }
        }
      }
    } else {
      weight = 1.0E-100;
    }
    p.weight = weight;

    weights.push_back(weight);
    p = SetAssociations(p, assoc, x_sense, y_sense);
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());
  std::vector<Particle> resampled;
  for (int i = 0; i < num_particles; i++) {
    resampled.push_back(particles[distribution(gen)]);
  }
  particles.clear();
  particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
