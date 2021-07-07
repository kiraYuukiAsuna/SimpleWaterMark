// 2021 07 03
// 2019213336WRL

#include "main.h"

#include <iostream>
#include <string>

#include <Windows.h>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#pragma execution_character_set( "utf-8" )

#ifdef _DEBUG
#pragma comment(lib, "opencv_world451d.lib")
#else
#pragma comment(lib, "opencv_world451.lib")
#endif


cv::Mat addWaterMark(cv::Mat& src, std::string& waterMarkText, double& position);
cv::Mat addWaterMarkRGB(cv::Mat& src, std::string& waterMarkText, double& position, bool allChannel);
cv::Mat addWaterMarkYUV(cv::Mat& src, std::string& waterMarkText, double& position);
cv::Mat getWaterMark(cv::Mat& src);
std::vector<cv::Mat> getWaterMarkRGB(cv::Mat& src);
std::vector<cv::Mat> getWaterMarkYUV(cv::Mat& src);
cv::Mat getTransposeImage(const cv::Mat& input);

int main(int argc, char* argv[]) {
	SetConsoleOutputCP(65001); // chcp 65001

	boost::program_options::options_description parameters_defination("Ƶ��ˮӡ�����(Buy 2019213336WRL):");

	parameters_defination.add_options()
		("input,i", boost::program_options::value<std::string>(), "������ļ� Example: video.mp4")
		("output,o", boost::program_options::value<std::string>()->default_value("output001.avi"), "������ļ� Example: output001.avi")
		("watermark,w", boost::program_options::value<std::string>()->default_value("2019213336WRL"), "ˮӡ����")
		("type,t", boost::program_options::value<std::string>()->default_value("video"), "������������,ͼƬ������Ƶ")
		("dump,d", boost::program_options::value<bool>()->default_value(false), "���JPG����")
		("position,p", boost::program_options::value<double>()->default_value(0.2), "ˮӡλ�ã�Ĭ�� 0.2")
		("frame,f", boost::program_options::value<int>(), "ָ������֡����")
		("yuv,v", "YUVģʽ")
		("mode,m", boost::program_options::value<std::string>(), "����ģʽ[Decode/Encode]")
		("help,H", "��ʾ����")
		;

	boost::program_options::variables_map parameters_map;

	try {
		boost::program_options::store(boost::program_options::parse_command_line(argc, argv, parameters_defination), parameters_map);
		boost::program_options::notify(parameters_map);
	}
	catch (...) {
		std::cerr << "������δ�������!\n";
	}

	if (parameters_map.count("help")) {
		std::cout << parameters_defination << std::endl;
	}

	if (parameters_map.count("input")) {
		std::string inputFile = parameters_map["input"].as<std::string>();

		if (parameters_map.count("mode")) {

			if (parameters_map.count("yuv")) {
				std::cout << "YUVģʽ\n";
			}
			else {
				std::cout << "RGBģʽ\n";
			}


			std::string mode = parameters_map["mode"].as<std::string>();
			if (boost::iequals(mode, "decode")) {
				std::cout << "����ģʽ\n";

				cv::Mat src;

				// ����ͼƬ
				src = cv::imread(inputFile, cv::IMREAD_COLOR);

				if (src.empty()) {
					fprintf(stderr, "ͼƬ��ȡʧ��%s\n", inputFile.c_str());
					return 1;
				}

				std::vector<cv::Mat> result;
				if (parameters_map.count("yuv")) {
					result = getWaterMarkYUV(src);
				}
				else {
					result = getWaterMarkRGB(src);
				}

				src.release();

				std::string outfile = inputFile + "_decoded_";
				for (int i = 0; i < result.size(); i++) {
					cv::imwrite(outfile + std::to_string(i) + ".jpg", result[i]);
					fprintf(stderr, "�ļ��ѱ���%s\n", outfile.c_str());
				}


				return 0;
			}
			else {
				std::string waterMarkText = "2019213336WRL";

				if (parameters_map.count("watermark")) {
					waterMarkText = parameters_map["watermark"].as<std::string>();
				}

				double position = 0.2;
				if (parameters_map.count("position")) {
					position = parameters_map["position"].as<double>();
				}

				std::string type = parameters_map["type"].as<std::string>();
				if (boost::iequals(type, "pic")) {
					std::cout << "ͼƬ����ģʽ\n";

					std::string outputPath = "output001.jpg";

					if (parameters_map.count("output")) {
						outputPath = parameters_map["output"].as<std::string>();
					}

					cv::Mat src;
					src = cv::imread(inputFile);

					cv::Mat result;

					if (parameters_map.count("yuv")) {
						result = addWaterMarkYUV(src, waterMarkText, position);
					}
					else {
						result = addWaterMarkRGB(src, waterMarkText, position, true);
					}

					cv::imwrite(outputPath, result);
					std::cout << "�ļ��ѱ���%s\n" << outputPath << std::endl;

					return 0;
				}
				else {
					std::cout << "��Ƶ����ģʽ\n";

					cv::VideoCapture inputVideo(inputFile);
					if (!inputVideo.isOpened()) {
						std::cout << "��Ƶ��ȡʧ��%s\n" << inputFile << std::endl;
						return -1;
					}

					int totalFrameCount = inputVideo.get(cv::CAP_PROP_FRAME_COUNT);
					int currentFrameCount = 0;

					std::string outputPath = "output001.avi";

					if (parameters_map.count("output")) {
						outputPath = parameters_map["output"].as<std::string>();
					}

					cv::VideoWriter outputVideo;
					outputVideo.open(outputPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), inputVideo.get(cv::CAP_PROP_FPS), cv::Size(inputVideo.get(cv::CAP_PROP_FRAME_WIDTH), inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT)), true);

					if (!outputVideo.isOpened()) {
						std::cout << "�޷�д����Ƶ\n";
						return -1;
					}

					cv::Mat frame;

					bool dumpFrame = false;
					if (parameters_map.count("dump")) {
						dumpFrame = parameters_map["dump"].as<bool>();
					}

					bool limitFrame = false;
					int limitFrameCount = 0;
					if (parameters_map.count("frame")) {
						limitFrame = true;
						limitFrameCount = parameters_map["frame"].as<int>();
					}

					bool endWhileLoopFlage = true;

					while (endWhileLoopFlage) {
						inputVideo >> frame;

						if (frame.empty()) {
							std::cout << "�������!";
							break;
						}

						cv::Mat result;

						if (parameters_map.count("yuv")) {
							result = addWaterMarkYUV(frame, waterMarkText, position);
						}
						else {
							result = addWaterMarkRGB(frame, waterMarkText, position, true);
						}

						outputVideo << result;

						if (dumpFrame) {
							std::string outfile = "frame" + std::to_string(currentFrameCount) + ".jpg";
							cv::imwrite(outfile, result);
							std::cout << "�ļ��ѱ���%s\n" << outfile << std::endl;
						}

						fprintf(stderr, "Frame %d / %d\r", currentFrameCount, totalFrameCount);

						currentFrameCount++;
						if (currentFrameCount >= totalFrameCount) {
							break
								;
						}
						if (limitFrame && currentFrameCount >= limitFrameCount) {
							break;
						}
					}

					inputVideo.release();
					outputVideo.release();

					cv::destroyAllWindows();

					return 0;
				}
			}
		}
		else {
			std::cout << "û��ָ������ģʽ\n";
			return -1;
		}
	}
	else {
		std::cout << "û��ָ�������ļ�\n";
		return -1;
	}
	return 0;
}

cv::Mat addWaterMarkYUV(cv::Mat& src, std::string& waterMarkText, double& position) {
	std::vector<cv::Mat> src_channels;
	cv::Mat srcYUV;

	cv::Mat targetYUV;
	cv::Mat targetBGR;
	std::vector<cv::Mat> target_channels;

	cv::cvtColor(src, srcYUV, cv::COLOR_BGR2YCrCb);

	cv::split(srcYUV, src_channels);

	// �ֱ�������ͨ��
	int i = 0;
	for (auto iter : src_channels) {
		if (i == 0) {
			target_channels.push_back(addWaterMark(src_channels[i], waterMarkText, position));
		}
		else {
			target_channels.push_back(src_channels[i]);
		}
		i++;
	}
	cv::merge(target_channels, targetYUV);
	cv::cvtColor(targetYUV, targetBGR, cv::COLOR_YCrCb2BGR);
	return targetBGR;
}

cv::Mat addWaterMarkRGB(cv::Mat& src, std::string& waterMarkText, double& position, bool allChannel)
{
	int col = src.cols;
	int row = src.rows;
	std::vector<cv::Mat> src_channels;
	cv::Mat target;
	std::vector<cv::Mat> target_channels;

	if (row > col) {
		cv::split(getTransposeImage(src), src_channels);
	}
	else {
		cv::split(src, src_channels);
	}

	// �ֱ�������ͨ��
	for (int i = 0; i < src_channels.size(); i++) {
		if (1) {
			target_channels.push_back(addWaterMark(src_channels[i], waterMarkText, position));
		}
		else {
			target_channels.push_back(src_channels[i]);
		}
	}
	cv::merge(target_channels, target);
	if (row > col) {
		return getTransposeImage(target);
	}
	return target;

}

cv::Mat getTransposeImage(const cv::Mat& input)
{
	cv::Mat result;
	transpose(input, result);
	return result;
}

cv::Mat addWaterMark(cv::Mat& src, std::string& waterMarkText, double& position) {
	// Ѱ�����ʺϵ�DFT��С
	int m = cv::getOptimalDFTSize(src.rows);
	int n = cv::getOptimalDFTSize(src.cols);

	cv::Mat padded; // ������ͼ�����
	cv::Mat magnitudeImage;

	// ������Ƭ��С����ˮӡ��С���
	double textSize = 0.0;
	int textWidth = 0;

	int minImgSize = src.rows > src.cols ? src.cols : src.rows;

	if (minImgSize < 150)
	{
		textSize = 1.0;
		textWidth = 1;
	}
	else if (minImgSize >= 150 && minImgSize < 300)
	{
		textSize = 1.5;
		textWidth = 2;
	}
	else if (minImgSize >= 300 && minImgSize < 400)
	{
		textSize = 2.5;
		textWidth = 3;
	}
	else if (minImgSize >= 400 && minImgSize < 650)
	{
		textSize = 5.0;
		textWidth = 5;
	}
	else if (minImgSize >= 650 && minImgSize < 1000)
	{
		textSize = 6.0;
		textWidth = 6;
	}
	else if (minImgSize >= 1000)
	{
		textSize = 7.5;;
		textWidth = 7;
	}

	cv::copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	// ����һ����Ϊ��Ž��,ת��Ϊfloat��
	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(),CV_32F) };
	cv::Mat complete;
	// �ϲ��ɶ��ͨ����ͼ����һ��ͨ��Ϊ�ղŸ��Ƶ�Ŀ�꣬�ڶ���Ϊ0
	cv::merge(planes, 2, complete);

	// ��ɢ����Ҷ�任
	cv::dft(complete, complete);


	double minv = 0.0, maxv = 0.0;
	double* minp = &minv;
	double* maxp = &maxv;
	// �������ֵ����Сֵ
	cv::minMaxIdx(complete, minp, maxp);
	//fprintf(stderr, "minv %.6f maxv %.6f\n", minv, maxv);

	// ����ƽ��ֵ������ˮӡǿ��
	int meanvalue = cv::mean(complete)[0];
	int watermark_scale = 0;
	//fprintf(stderr, "mean %d\n", meanvalue);
	if (meanvalue > 128)
	{
		watermark_scale = -log(abs(minv));
	}
	else
	{
		watermark_scale = log(abs(maxv));
	}
	//fprintf(stderr, "watermark_scale %.6f\n", watermark_scale);
	auto font = cv::FONT_HERSHEY_PLAIN;
	cv::Scalar color(watermark_scale, watermark_scale, watermark_scale);
	cv::Point pos(src.cols * position, src.rows * position);
	cv::putText(complete, waterMarkText, pos, cv::FONT_HERSHEY_PLAIN, textSize, color, textWidth);
	cv::flip(complete, complete, -1);
	cv::putText(complete, waterMarkText, pos, cv::FONT_HERSHEY_PLAIN, textSize, color, textWidth);
	cv::flip(complete, complete, -1);

	// �渵��Ҷ�任
	idft(complete, complete);

	split(complete, planes);

	magnitude(planes[0], planes[1], planes[0]);
	cv::Mat result = planes[0];
	result = result(cv::Rect(0, 0, src.cols, src.rows));
	// ��׼��
	normalize(result, result, 0, 1, cv::NORM_MINMAX);

	padded.release();
	magnitudeImage.release();
	planes[1].release();
	cv::Mat out;
	// ת��u_char�͵�Mat
	result.convertTo(out, CV_8U, 255.0);
	result.release();
	return out;
}

std::vector<cv::Mat> getWaterMarkRGB(cv::Mat& src)
{
	std::vector<cv::Mat> src_channels;
	cv::Mat target;
	std::vector<cv::Mat> target_channels;
	cv::split(src, src_channels);
	// �ֱ�������ͨ��
	for (int i = 0; i < src_channels.size(); i++) {
		target_channels.push_back(getWaterMark(src_channels[i]));
	}
	return target_channels;
}

std::vector<cv::Mat> getWaterMarkYUV(cv::Mat& src)
{
	std::vector<cv::Mat> src_channels;
	std::vector<cv::Mat> target_channels;
	cv::Mat srcYUV;
	cv::cvtColor(src, srcYUV, cv::COLOR_BGR2YCrCb);

	cv::split(srcYUV, src_channels);
	// �ֱ�������ͨ��
	for (int i = 0; i < src_channels.size(); i++) {
		if (i == 0) {
			target_channels.push_back(getWaterMark(src_channels[i]));
		}
	}
	return target_channels;
}

cv::Mat getWaterMark(cv::Mat& src) {
	int m = cv::getOptimalDFTSize(src.rows);
	int n = cv::getOptimalDFTSize(src.cols);

	cv::Mat padded; // ������ͼ�����
	cv::Mat magnitudeImage;

	double textSize = 1.5;
	int textWidth = 2;

	cv::copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	//����һ����Ϊ��Ž��,ת��Ϊfloat��
	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(),CV_32F) };
	cv::Mat complete;
	//�ϲ��ɶ��ͨ����ͼ����һ��ͨ��Ϊ�ղŸ��Ƶ�Ŀ�꣬�ڶ���Ϊ0
	cv::merge(planes, 2, complete);

	//��ɢ����Ҷ�任
	cv::dft(complete, complete);

	//����ʵ�����鲿
	split(complete, planes);
	magnitude(planes[0], planes[1], planes[0]);
	magnitudeImage = planes[0];

	//��������
	magnitudeImage += cv::Scalar::all(1);
	log(magnitudeImage, magnitudeImage);

	//�ü�
	magnitudeImage = magnitudeImage(cv::Rect(0, 0, src.cols, src.rows));

	//��׼����0~1
	normalize(magnitudeImage, magnitudeImage, 0, 1, cv::NORM_MINMAX);

	int cx = magnitudeImage.cols / 2;
	int cy = magnitudeImage.rows / 2;
	cv::Mat temp;
	cv::Mat q0(magnitudeImage, cv::Rect(0, 0, cx, cy));
	cv::Mat q1(magnitudeImage, cv::Rect(cx, 0, cx, cy));
	cv::Mat q2(magnitudeImage, cv::Rect(0, cy, cx, cy));
	cv::Mat q3(magnitudeImage, cv::Rect(cx, cy, cx, cy));
	q0.copyTo(temp);
	q3.copyTo(q0);
	temp.copyTo(q3);
	q1.copyTo(temp);
	q2.copyTo(q1);
	temp.copyTo(q2);

	cv::Mat out;
	// ת��u_char�͵�Mat
	magnitudeImage.convertTo(out, CV_8UC3, 255.0);
	magnitudeImage.release();
	return out;
}
