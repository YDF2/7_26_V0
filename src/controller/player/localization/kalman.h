#ifndef __KALMAN_H
#define __KALMAN_H

#include <iostream>
#include <limits.h>
#include <list>
#include <vector>
#include "math/math.hpp"
#include "soccermap.h"
#include "model.hpp"
#define Nn 4

class KA
{
public:
  
  struct State 
  {
    float x, y;
    float dir;
    State& operator=(const State& s)
    {
      x=s.x;
      y=s.y;
      dir=s.dir;
      return *this;
    }
  };

  struct Observe 
  {
    float x,y;
  };
  filed_info field_;
  State last_state;
  State now_state;
  Observe Z1;
  Observe Z2[2];
  Observe ZT1;
  Observe ZT2[2];
  float P[2][2];
  float H1[2][2];
  float K1[2][2];
  float H2[4][2];
  float K2[2][4];
  float Q[2][2];
  float R1[2][2];
  float R2[4][4];
  float detq;
  float detr;
  Eigen::Vector2f doublepost[2];
  float dirErr[4];
 
private:
  void setzero()
  {
     for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	Q[i][j]=0;
      
     for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	R1[i][j]=0;
      
     for(int i=0;i<4;i++)
      for(int j=0;j<4;j++)
	R2[i][j]=0;
     
     for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	P[i][j]=0;
  };
  void addPQ()
  {
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	P[i][j]=P[i][j]+Q[i][j];
  };
  void setq(float dat)
  {
    for(int i=0;i<2;i++)
      Q[i][i]=dat;
  };
  void setr(float dat)
  {
    for(int i=0;i<2;i++)
      R1[i][i]=0.01*dat;
    for(int i=0;i<4;i++)
      R2[i][i]=dat;
  };  
  
  void getH1(float M, float N1);
  void getH2(float M, float N1, int flag);
  void getK1();
  bool getK2();
  float getA(float arcs[Nn][Nn],int n);
  void getAStart(float arcs[Nn][Nn],int n,float ans[Nn][Nn]); 
  bool GetMatrixInverse(float src[Nn][Nn],int n,float des[Nn][Nn]);
  bool checksign(float a, float b);
  
public:
  void init();
  void forecast(const player_info &player_info_);
  int goalPostUpdate(const std::vector< goal_post > & posts_);
  //bool fieldLineUpdate(const list< sm::Line > &fieldlines);
  State obeupdate1();
  State obeupdate2();
  void drawRoboOnSoccerMap();
  void setPzero();
};


#endif