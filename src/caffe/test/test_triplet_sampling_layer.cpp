#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

	template <typename TypeParam>
	class TripletSamplingLayerTest: public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;
	protected:
		TripletSamplingLayerTest():
			blob_bottom_data_i_(new Blob<Dtype>(512, 2, 1, 1)),
			blob_bottom_data_j_(new Blob<Dtype>(512, 1, 1, 1)),
			//fill the values
			blob_top_anchors_(new Blob<Dtype>()),
			blob_top_positives_(new Blob<Dtype>()),
			blob_top_negatives_(new Blob<Dtype>())
			{
				FillerParameter filler_param;
				filler_param.set_min(-1.0);
				filler_param.set_max(1.0);
				UniformFiller<Dtype> filler(filler_param);
				filler.Fill(this->blob_bottom_data_i_);
				for(int i = 0; i < 512; i++)
				{
					Dtype* data = blob_bottom_data_j_->mutable_cpu_data();
				    for (int i = 0; i < 512; ++i) {
				      	data[i] = rand() % 2;
    				}
				}
				blob_bottom_vec_.push_back(blob_bottom_data_i_);
				blob_bottom_vec_.push_back(blob_bottom_data_j_);
				blob_top_vec_.push_back(blob_top_anchors_);
				blob_top_vec_.push_back(blob_top_positives_);
				blob_top_vec_.push_back(blob_top_negatives_);
				cout << "init finished" << endl;
			}
			
		virtual ~TripletSamplingLayerTest() {
			delete blob_bottom_data_i_;
			delete blob_bottom_data_j_;
			delete blob_top_anchors_;
			delete blob_top_positives_;
			delete blob_top_negatives_;
		}

	Blob<Dtype>*  blob_bottom_data_i_;
	Blob<Dtype>*  blob_bottom_data_j_;
	Blob<Dtype>*  blob_top_anchors_;
	Blob<Dtype>*  blob_top_positives_;
	Blob<Dtype>*  blob_top_negatives_;
	vector<Blob<Dtype>*> blob_bottom_vec_;
	vector<Blob<Dtype>*> blob_top_vec_;
	};
	TYPED_TEST_CASE(TripletSamplingLayerTest, TestDtypesAndDevices);
	TYPED_TEST(TripletSamplingLayerTest, TestForward){
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		TripletSamplingLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		cout << "start forward" << endl;
		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		cout << "end forward" << endl;
		//manually compute to compare
	}
	TYPED_TEST(TripletSamplingLayerTest, TestGradient){
//		typedef typename TypeParam::Dtype Dtype;
//		LayerParameter layer_param;
//		TripletLossLayer<Dtype> layer(layer_param);
//		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//		GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
		//check the gradient for the first two bottom layers.
//		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
//		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 1);
	}
}//namespace caffe.