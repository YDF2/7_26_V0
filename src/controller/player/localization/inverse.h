#include <iostream>
#include <cmath>
#include <ctime>

#define N 4

using namespace std;

//矩阵乘法
float * mul(float A[N*N],float B[N*N])
{
    float *C=new float[N*N]{};
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            for(int k=0;k<N;k++)
            {
                C[i*N+j] += A[i*N+k]*B[k*N+j];
            }
        }
    }

    //若绝对值小于10^-10,则置为0（这是我自己定的）
    for(int i=0;i<N*N;i++)
    {
        if(abs(C[i])<pow(10,-10))
        {
            C[i]=0;
        }
    }

    return C;
}

//LUP分解
bool LUP_Descomposition(float A[N*N],float L[N*N],float U[N*N],int P[N])
{
    int row=0;
    for(int i=0;i<N;i++)
    {
        P[i]=i;
    }
    for(int i=0;i<N-1;i++)
    {
        float p=0.0f;
        for(int j=i;j<N;j++)
        {
            if(abs(A[j*N+i])>p)
            {
                p=abs(A[j*N+i]);
                row=j;
            }
        }
        if(0==p)
        {
            cout<< "矩阵奇异，无法计算逆" <<endl;
            return false;
        }

        //交换P[i]和P[row]
        int tmp=P[i];
        P[i]=P[row];
        P[row]=tmp;

        float tmp2=0.0f;
        for(int j=0;j<N;j++)
        {
            //交换A[i][j]和 A[row][j]
            tmp2=A[i*N+j];
            A[i*N+j]=A[row*N+j];
            A[row*N+j]=tmp2;
        }

        //以下同LU分解
        float u=A[i*N+i],l=0.0f;
        for(int j=i+1;j<N;j++)
        {
            l=A[j*N+i]/u;
            A[j*N+i]=l;
            for(int k=i+1;k<N;k++)
            {
                A[j*N+k]=A[j*N+k]-A[i*N+k]*l;
            }
        }

    }

    //构造L和U
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<=i;j++)
        {
            if(i!=j)
            {
                L[i*N+j]=A[i*N+j];
            }
            else
            {
                L[i*N+j]=1;
            }
        }
        for(int k=i;k<N;k++)
        {
            U[i*N+k]=A[i*N+k];
        }
    }
    return true;

}

//LUP求解方程
float * LUP_Solve(float L[N*N],float U[N*N],int P[N],float b[N])
{
    float *x=new float[N]();
    float *y=new float[N]();

    //正向替换
    for(int i = 0;i < N;i++)
    {
        y[i] = b[P[i]];
        for(int j = 0;j < i;j++)
        {
            y[i] = y[i] - L[i*N+j]*y[j];
        }
    }
    //反向替换
    for(int i = N-1;i >= 0; i--)
    {
        x[i]=y[i];
        for(int j = N-1;j > i;j--)
        {
            x[i] = x[i] - U[i*N+j]*x[j];
        }
        x[i] /= U[i*N+i];
    }
    return x;
}

/*****************矩阵原地转置BEGIN********************/

/* 后继 */
int getNext(int i, int m, int n)
{
  return (i%n)*m + i/n;
}

/* 前驱 */
int getPre(int i, int m, int n)
{
  return (i%m)*n + i/m;
}

/* 处理以下标i为起点的环 */
void movedata(float *mtx, int i, int m, int n)
{
  float temp = mtx[i]; // 暂存
  int cur = i;    // 当前下标
  int pre = getPre(cur, m, n);
  while(pre != i)
  {
    mtx[cur] = mtx[pre];
    cur = pre;
    pre = getPre(cur, m, n);
  }
  mtx[cur] = temp;
}

/* 转置，即循环处理所有环 */
void transpose(float *mtx, int m, int n)
{
  for(int i=0; i<m*n; ++i)
  {
    int next = getNext(i, m, n);
    while(next > i) // 若存在后继小于i说明重复,就不进行下去了（只有不重复时进入while循环）
      next = getNext(next, m, n);
    if(next == i)  // 处理当前环
      movedata(mtx, i, m, n);
  }
}
/*****************矩阵原地转置END********************/

//LUP求逆(将每列b求出的各列x进行组装)
bool LUP_solve_inverse(float A[N][N],float inv[N][N])
{
    //创建矩阵A的副本，注意不能直接用A计算，因为LUP分解算法已将其改变
    float *A_mirror = new float[N*N]();
    float *inv_A=new float[N*N]();//最终的逆矩阵（还需要转置）
    float *inv_A_each=new float[N]();//矩阵逆的各列
    //double *B    =new double[N*N]();
    float *b    =new float[N]();//b阵为B阵的列矩阵分量

    for(int i=0;i<N;i++)
    {
        float *L=new float[N*N]();
        float *U=new float[N*N]();
        int *P=new int[N]();

        //构造单位阵的每一列
        for(int i=0;i<N;i++)
        {
            b[i]=0;
        }
        b[i]=1;

        //每次都需要重新将A复制一份
        for(int i=0;i<N;i++)
	  for(int j=0;j<N;j++)
          {
             A_mirror[i*N+j]=A[i][j];
          }

        if(!LUP_Descomposition(A_mirror,L,U,P))
            return false;

        inv_A_each=LUP_Solve (L,U,P,b);
        memcpy(inv_A+i*N,inv_A_each,N*sizeof(float));//将各列拼接起来
    }
    transpose(inv_A,N,N);//由于现在根据每列b算出的x按行存储，因此需转置
    
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++)
	inv[i][j]=inv_A[i*N+j];

    return true;
	
    //return inv_A;
}

