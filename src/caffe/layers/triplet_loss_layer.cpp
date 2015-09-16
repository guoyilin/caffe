/**
* Triplet loss layer
*/

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template <typename Dtype>
	void TripletLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		CHECK_EQ(bottom[0]->num(), bottom[1]->num());
		CHECK_EQ(bottom[1]->num(), bottom[2]->num());
		CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
		CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());
		CHECK_EQ(bottom[0]->height(), 1);//fc layer: height & width = 1
		CHECK_EQ(bottom[0]->width(), 1);
		CHECK_EQ(bottom[1]->height(), 1);
		CHECK_EQ(bottom[1]->width(), 1);
		CHECK_EQ(bottom[2]->width(), 1);
		CHECK_EQ(bottom[2]->height(), 1);

//		CHECK_EQ(bottom[3]->channels(), 1);
//		CHECK_EQ(bottom[3]->height(), 1);
//		CHECK_EQ(bottom[3]->width(), 1);

		diff_ap_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
		diff_an_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
		diff_pn_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);

		diff_sq_ap_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
		diff_sq_an_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
		dist_sq_ap_.Reshape(bottom[0]->num(), 1, 1, 1);
		dist_sq_an_.Reshape(bottom[0]->num(), 1, 1, 1);

		summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
		for(int i = 0 ; i < bottom[0]->channels(); ++i)
		{
			summer_vec_.mutable_cpu_data()[i] = Dtype(1);
		}
		dist_binary_.Reshape(bottom[0]->num(), 1, 1, 1);
		for(int i =0 ; i < bottom[0]->num(); ++i){
			dist_binary_.mutable_cpu_data()[i] = Dtype(1);
		}
	}

	template <typename Dtype>
	void TripletLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*> & bottom, const vector<Blob<Dtype>*> & top){
		int count = bottom[0]->count();//count= n * c * h * w
//		const Dtype* sampleW = bottom[3]->cpu_data(); // 1
		caffe_sub(
			count, 
			bottom[0]->cpu_data(), // a
			bottom[1]->cpu_data(), //p
			diff_ap_.mutable_cpu_data()); // diff_ap_= a - p
		caffe_sub(
			count,
			bottom[0]->cpu_data(), //a
			bottom[2]->cpu_data(), //n
			diff_an_.mutable_cpu_data()); // diff_an_ = a - n
		caffe_sub(
			count, 
			bottom[1]->cpu_data(), //p
			bottom[2]->cpu_data(), //n
			diff_pn_.mutable_cpu_data() // diff_pn_ = p - n
			);
		const int channels = bottom[0]->channels();
		Dtype margin = this->layer_param_.triplet_loss_param().margin();// alpha
		Dtype loss(0.0); //record the  loss of this batch.
		for(int i = 0; i < bottom[0]->num(); ++i) {//for all triplet
			dist_sq_ap_.mutable_cpu_data()[i] = caffe_cpu_dot(
				channels, diff_ap_.cpu_data() + (i*channels), diff_ap_.cpu_data() + (i * channels));
			dist_sq_an_.mutable_cpu_data()[i] = caffe_cpu_dot(
				channels, diff_an_.cpu_data() + (i * channels), diff_an_.cpu_data() + (i * channels));
			//mdist= one triplet loss
			Dtype mdist =  std::max(margin + dist_sq_ap_.cpu_data()[i] - dist_sq_an_.cpu_data()[i], Dtype(0.0));
			loss += mdist;
			if(mdist == Dtype(0)){
				caffe_set(channels, Dtype(0), diff_ap_.mutable_cpu_data() + (i * channels));
				caffe_set(channels, Dtype(0), diff_an_.mutable_cpu_data() + (i * channels));
				caffe_set(channels, Dtype(0), diff_pn_.mutable_cpu_data() + (i * channels));
			}
		}
		loss = loss/static_cast<Dtype>(bottom[0]->num())/Dtype(2);
		top[0]->mutable_cpu_data()[0] = loss;
	}

	template<typename Dtype>
	void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, 
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//		const Dtype* sampleW = bottom[3]->cpu_data();
		//for 3 bottom data
		for(int i = 0; i < 3; ++i)
		{
			if(propagate_down[i]){
				const Dtype sign = (i < 2) ? -1: 1;
				const Dtype alpha = sign * top[0]->cpu_diff()[0] / static_cast<Dtype>(bottom[i]->num());
				int num = bottom[i]->num();
				int channels = bottom[i]->channels();
				for(int j = 0; j < num; ++j)//for each image
				{
					Dtype* bout = bottom[i]->mutable_cpu_diff();
					if(i == 0){//anchor
						caffe_cpu_axpby(
							channels, 
							alpha,// * sampleW[j], 
							diff_pn_.cpu_data() + (j * channels),
							Dtype(0.0),
							bout + (j*channels)
							);	
					}else if(i == 1) { //positive
						caffe_cpu_axpby(
							channels,
							alpha, //*sampleW[j],
							diff_ap_.cpu_data() + (j*channels),
							Dtype(0.0),
							bout + (j*channels)
							);
					}else if(i==2){ //negative
						caffe_cpu_axpby(
							channels, 
							alpha, // * sampleW[j],
							diff_an_.cpu_data() + (j*channels),
							Dtype(0.0),
							bout + (j*channels)
							);
					}
				}
			}
		}
	}
#ifdef CPU_ONLY
	STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}//namespace caffe