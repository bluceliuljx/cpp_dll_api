#ifndef PL_TYPE_H_
#define PL_TYPE_H_

#include <vector>
#include <string>
/************************************************************************/
/*                        basic structures                              */
/************************************************************************/

//outputs
typedef struct OutParams {
	int _bbX1;
	int _bbY1;
	int _bbX2;
	int _bbY2;
	float _confidence;
	int _id;
	int _img_idx;
}OutParams;


#endif //PL_TYPE_H_