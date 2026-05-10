#include "kalman.h"
#include <stdio.h>
#include "inverse.h"

using namespace std;
using namespace robot_math;
using namespace Eigen;

void KA::init()
{
  now_state.x=SOCCERMAP->width() - SOCCERMAP->center().x();
  now_state.y=SOCCERMAP->height() - SOCCERMAP->center().y();
  now_state.dir=0;
  detq=exp(-20);
  detr=exp(-20);
  setzero();
  setq(detq);
  setr(detr);         
}

void KA::setPzero()
{
  for(int i=0;i<2;i++)
    for(int j=0;j<2;j++)
      P[i][j]=0;
    
  addPQ();
}


void KA::forecast(const player_info &player_info_)
{ 
  now_state.x= player_info_.x*100.0;
  now_state.y= player_info_.y*100.0;
  now_state.dir= player_info_.dir;
  addPQ();
}



int KA::goalPostUpdate(const vector< goal_post > & posts_)
{
     int size=posts_.size();
     int flag=0;  
     for(const goal_post & post: posts_)
    {
        vector<Vector2f> postPosL, postPosR; 

        if (post.side == goal_post::SENSORMODEL_SIDE_OUR)
        {
            postPosL.push_back(SOCCERMAP->flagPos().at(SoccerMap::FLAG_DOOR_OUR_N));
            postPosR.push_back(SOCCERMAP->flagPos().at(SoccerMap::FLAG_DOOR_OUR_P));
        }
        else if (post.side == goal_post::SENSORMODEL_SIDE_OPP)
        {
            postPosL.push_back(SOCCERMAP->flagPos().at(SoccerMap::FLAG_DOOR_OPP_P));
            postPosR.push_back(SOCCERMAP->flagPos().at(SoccerMap::FLAG_DOOR_OPP_N));
        }
        else
        {
            postPosL.push_back(SOCCERMAP->flagPos().at(SoccerMap::FLAG_DOOR_OUR_N));
            postPosR.push_back(SOCCERMAP->flagPos().at(SoccerMap::FLAG_DOOR_OUR_P));
            postPosR.push_back(SOCCERMAP->flagPos().at(SoccerMap::FLAG_DOOR_OPP_N));
            postPosL.push_back(SOCCERMAP->flagPos().at(SoccerMap::FLAG_DOOR_OPP_P));
        }

        float maxWeight = 0.0f;
        vector<Vector2f> postPos;

        if (post.type == goal_post::SENSORMODEL_POST_L)
        {
          postPos.insert(postPos.end(), postPosL.begin(), postPosL.end());
        }
        else if (post.type == goal_post::SENSORMODEL_POST_R)
        {
          postPos.insert(postPos.end(), postPosR.begin(), postPosR.end());
        }
        else //unknown
        {
          postPos.insert(postPos.end(), postPosL.begin(), postPosL.end());          
          postPos.insert(postPos.end(), postPosR.begin(), postPosR.end());
        }
           
        float bearErr, distErr;
	      float minErr=1000000;
	      Vector2f pos(now_state.x, now_state.y);
        Vector2f pppp;
        for(const Vector2f & fp: postPos)
	      {
	          bearErr = normalize_deg(azimuth_deg(fp - pos)-now_state.dir-post._theta);
            distErr = (fp - pos).norm() - post._distance;
	          // cout<<fp.x()<<"  "<<fp.y()<<"  "<<abs(bearErr)<<"  "<<abs(distErr)<<endl;
	          // cout<<(fp - pos).angle()<<"  "<<now_state.dir<<"  "<<post._theta<<endl;
	          if(20*abs(bearErr)+abs(distErr)<minErr)// && abs(normalize_deg(azimuth(fp - pos)-now_state.dir))<=50)
	          {
	            minErr=20*abs(bearErr)+abs(distErr);
	            float M=fp.y()-now_state.y;
              float N1=fp.x()-now_state.x;
	            if(size==1)
	            {
                pppp=fp;
	              doublepost[0].x()=fp.x();
		            doublepost[0].y()=fp.y();
	              float delta=(post._theta+now_state.dir)/180*3.1415926;
	              Z1.x=post._distance*cos(delta);
		            Z1.y=post._distance*sin(delta);
		            ZT1.x=N1;
		            ZT1.y=M;
		            getH1(M,N1);
	            }
	            else if(size==2)
	            {
		            doublepost[flag].x()=fp.x();
	      	      doublepost[flag].y()=fp.y();
                Z2[flag].x=(post._distance*cos((now_state.dir+post._theta)/180*3.1415926))*(post._distance*cos((now_state.dir+post._theta)/180*3.1415926));
                Z2[flag].y=(post._distance*sin((now_state.dir+post._theta)/180*3.1415926))*(post._distance*sin((now_state.dir+post._theta)/180*3.1415926));
                ZT2[flag].x=N1*N1;
                ZT2[flag].y=M*M;          
                getH2(M,N1,flag); 
	            }
	        }
	      }
        //LOG(LOG_INFO)<<pppp.transpose()<<endll;
	      flag++;
    }
    return size;
}

void KA::getH1(float M, float N1)
{
  float a=0.00000001;
  H1[0][0]=-1;         H1[0][1]=0;         
  H1[1][0]=0;             H1[1][1]=-1;
}

void KA::getH2(float M, float N1, int flag)
{
  H2[0+2*flag][0]=-2*N1;         H2[0+2*flag][1]=0;        
  H2[1+2*flag][0]=0;             H2[1+2*flag][1]=-2*M;
}

void KA::getK1()
{
  float ph[2][2]={0};
  for(int i=0;i<2;i++)
    for(int j=0;j<2;j++)
      for(int k=0;k<2;k++)
        ph[i][j]+=P[i][k]*H1[j][k];
   
  float hph[2][2]={0};
  for(int i=0;i<2;i++)
    for(int j=0;j<2;j++)
      for(int k=0;k<2;k++)
        hph[i][j]+=H1[i][k]*ph[k][j];
      
  for(int i=0;i<2;i++)
    for(int j=0;j<2;j++)
      hph[i][j]=hph[i][j]+R1[i][j];
    
  float ni[2][2]={0};
  float a = 1/(hph[0][0]*hph[1][1]-hph[1][0]-hph[0][1]);
  if(a!=0)
  {
    ni[0][0]=a*hph[1][1];     ni[0][1]=-a*hph[0][1];
    ni[1][0]=-a*hph[1][0];    ni[1][1]=a*hph[0][0];
  }
  
  for(int i=0;i<2;i++)
    for(int j=0;j<2;j++)
    {
      float b=0;
      for(int k=0;k<2;k++)
	b+=ph[i][k]*ni[k][j];    
      K1[i][j]=b;
    }
}

bool KA::getK2()
{
  float ph[2][4]={0};
  for(int i=0;i<2;i++)
    for(int j=0;j<4;j++)
      for(int k=0;k<2;k++)
        ph[i][j]+=P[i][k]*H2[j][k];
   
  float hph[4][4]={0};
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      for(int k=0;k<2;k++)
        hph[i][j]+=H2[i][k]*ph[k][j];
      
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      hph[i][j]=hph[i][j]+R2[i][j];
    
  float ni[4][4]={0};
   if(!LUP_solve_inverse(hph,ni))
   {
     return false;
   }
  /*
  float I[4][4];
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
    {
      float b=0;
      for(int k=0;k<4;k++)
	b+=hph[i][k]*ni[k][j];
      I[i][j]=b;
      cout<<b<<"  ";
    }
  */ 
 
  for(int i=0;i<2;i++)
    for(int j=0;j<4;j++)
    {
      float b=0;
      for(int k=0;k<4;k++)
	b+=ph[i][k]*ni[k][j];  
      K2[i][j]=b;
    }

  return true;
}

KA::State KA::obeupdate1()
{
  getK1();
  
  float kz[2]={0};
  for(int i=0;i<2;i++)
      kz[i]=K1[i][0]*(Z1.x-ZT1.x)
           +K1[i][1]*(Z1.y-ZT1.y);
  //cout<<Z1.x<<"          "<<ZT1.x<<endl;
  //cout<<Z1.y<<"          "<<ZT1.y<<endl;
  //cout<<now_state.x<<"  "<<now_state.y<<endl;
  if(kz[1]<100)
  {
    now_state.x+=kz[0];
    now_state.y+=kz[1];
  }
  //now_state.dir=normalizeAngle<AngDeg>(kz[2]+now_state.dir);
  //cout<<kz[0]<<"  "<<kz[1]<<endl;
  //cout<<now_state.x<<"  "<<now_state.y<<endl;
  
 
  
  float ikh[2][2];
  for(int i=0;i<2;i++)
    for(int j=0;j<2;j++)
    {
      float b=0;
      for(int k=0;k<2;k++)
	b+=K1[i][k]*H1[k][j];
      if(i==j)
	ikh[i][j]=1-b;
      else
	ikh[i][j]=-b;
    }
  for(int i=0;i<2;i++)
    for(int j=0;j<2;j++)
    {
      float b=0;
      for(int k=0;k<2;k++)
	b+=ikh[i][k]*P[k][j];
      P[i][j]=b;
    }
  return now_state;
}

KA::State KA::obeupdate2()
{
  if(!getK2())
    return now_state;
  float kz[2];
  for(int i=0;i<2;i++)
    kz[i]=K2[i][0]*(Z2[0].x-ZT2[0].x)
         +K2[i][1]*(Z2[0].y-ZT2[0].y)
	 +K2[i][2]*(Z2[1].x-ZT2[1].x)
	 +K2[i][3]*(Z2[1].y-ZT2[1].y);
  now_state.x+=kz[0];
  now_state.y+=kz[1];
 // now_state.dir=normalizeAngle<AngDeg>(kz[2]+now_state.dir);
  //cout<<kz[0]<<"    "<<kz[1]<<endl;
  //cout<<now_state.x<<"  "<<now_state.y<<endl;
  float ikh[2][2];
  for(int i=0;i<2;i++)
    for(int j=0;j<2;j++)
    {
      float b=0;
      for(int k=0;k<4;k++)
	b+=K2[i][k]*H2[k][j];
      if(i==j)
	ikh[i][j]=1-b;
      else
	ikh[i][j]=-b;
    }
  for(int i=0;i<2;i++)
    for(int j=0;j<2;j++)
    {
      float b=0;
      for(int k=0;k<2;k++)
	b+=ikh[i][k]*P[k][j];
      P[i][j]=b;
    }
    return now_state;
}

/*
void KA::drawRoboOnSoccerMap()
{
    if (!DEBUG_SOCCER_MAP.frameValid())
    {
        return ;
    }

    auto frame = DEBUG_SOCCER_MAP.mapImageFrame();
    DEBUG_SOCCER_MAP.lockImageFrame();

        imgproc::ImagePainter::drawCircle(frame,
                                          DEBUG_SOCCER_MAP.getFramePosition(Vector2i(now_state.x, now_state.y)),
                                          5, imgproc::BLACK);
        imgproc::ImagePainter::drawLine(frame,
                                        DEBUG_SOCCER_MAP.getFramePosition(Vector2i(now_state.x, now_state.x)),
                                        DEBUG_SOCCER_MAP.getFramePosition(Vector2i(now_state.x + 8.0f * cosDeg<AngDeg>(now_state.dir),
                                                now_state.y + 8.0f * sinDeg<AngDeg>(now_state.dir))),
                                        1, imgproc::BLACK);

    DEBUG_SOCCER_MAP.unlockImageFrame();
}
*/



float KA::getA(float arcs[Nn][Nn],int n)  
{  
    if(n==1)
    {
        return arcs[0][0];
    }
    float ans = 0;
    float temp[Nn][Nn]={0.0};
    int i,j,k;
    for(i=0;i<n;i++)
    {
        for(j=0;j<n-1;j++)
        {
            for(k=0;k<n-1;k++)
            {
                temp[j][k] = arcs[j+1][(k>=i)?k+1:k];
                
            }
        }
        double t = getA(temp,n-1);
        if(i%2==0)
        {
            ans += arcs[0][i]*t;
        }
        else
        {
            ans -=  arcs[0][i]*t;
        }
    }
    return ans;
}  

void KA::getAStart(float arcs[Nn][Nn],int n,float ans[Nn][Nn]) 
{  
    if(n==1)
    {
        ans[0][0] = 1;
        return;
    }
    int i,j,k,t;
    float temp[Nn][Nn];
    for(i=0;i<n;i++)
    {
        for(j=0;j<n;j++)
        {
            for(k=0;k<n-1;k++)
            {
                for(t=0;t<n-1;t++)
                {
                    temp[k][t] = arcs[k>=i?k+1:k][t>=j?t+1:t];
                }
            }
            
            
            ans[j][i]  =  getA(temp,n-1); 
            if((i+j)%2 == 1)
            {
                ans[j][i] = - ans[j][i];
            }
        }
    }
}  

bool KA::GetMatrixInverse(float src[Nn][Nn],int n,float des[Nn][Nn])
{
    float flag=getA(src,n);
    float t[Nn][Nn];
    if(0==flag)
    {
        cout<< "原矩阵行列式为0，无法求逆。请重新运行" <<endl;
        return false;//如果算出矩阵的行列式为0，则不往下进行
    }
    else
    {
        getAStart(src,n,t);
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n;j++)
            {
                des[i][j]=t[i][j]/flag;
            }
            
        }
    }
    
    return true;
}