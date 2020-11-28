#include "types/enum.hpp"


namespace darknet
{
    std::ostream& operator<< (std::ostream& out, const DataType& obj)
    {
        switch (obj)
        {
        case DataType::DT_FLOAT16:
            out << "float16";
            break;
        case DataType::DT_FLOAT32:
            out << "float32";
            break;
        case DataType::DT_FLOAT64:
            out << "float64";
            break;
        case DataType::DT_INT8:
            out << "int8";
            break;
        case DataType::DT_INT32:
            out << "int32";
            break;
        case DataType::DT_INT64:
            out << "int64";
            break;
        case DataType::DT_UINT8:
            out << "uint8";
            break;
        case DataType::DT_UINT32:
            out << "uint32";
            break;
        case DataType::DT_UINT64:
            out << "uint64";
            break;
        }
        return out;
    }

    std::ostream& operator<< (std::ostream& out, const DeviceType& obj)
    {
        switch (obj)
        {
        case DeviceType::GPU:
            out << "GPU";
            break;
        case DeviceType::CPU:
            out << "CPU";
            break;
        }

        return out;
    }

    std::ostream& operator<< (std::ostream& out, const LayerType& obj)
    {
        switch (obj)
        {
        case LayerType::CONVOLUTIONAL:
            out << "Convolutional";
            break;
        case LayerType::DECONVOLUTIONAL:
            out << "Deconvolutional";
            break;
        case LayerType::CONNECTED:
            out << "Connected";
            break;
        case LayerType::MAXPOOL:
            out << "Maxpool";
            break;
        case LayerType::LOCAL_AVGPOOL:
            out << "Local_avgpool";
            break;
        case LayerType::SOFTMAX:
            out << "Softmax";
            break;
        case LayerType::DETECTION:
            out << "Detection";
            break;
        case LayerType::DROPOUT:
            out << "Dropout";
            break;
        case LayerType::CROP:
            out << "Crop";
            break;
        case LayerType::ROUTE:
            out << "Route";
            break;
        case LayerType::COST:
            out << "Cost";
            break;
        case LayerType::NORMALIZATION:
            out << "Normalization";
            break;
        case LayerType::AVGPOOL:
            out << "Avgpool";
            break;
        case LayerType::LOCAL:
            out << "Local";
            break;
        case LayerType::SHORTCUT:
            out << "Shortcut";
            break;
        case LayerType::SCALE_CHANNELS:
            out << "Scale_channels";
            break;
        case LayerType::SAM:
            out << "Sam";
            break;
        case LayerType::ACTIVE:
            out << "Active";
            break;
        case LayerType::RNN:
            out << "Rnn";
            break;
        case LayerType::GRU:
            out << "Gru";
            break;
        case LayerType::LSTM:
            out << "Lstm";
            break;
        case LayerType::CONV_LSTM:
            out << "Conv_lstm";
            break;
        case LayerType::HISTORY:
            out << "History";
            break;
        case LayerType::CRNN:
            out << "Crnn";
            break;
        case LayerType::BATCHNORM:
            out << "Batchnorm";
            break;
        case LayerType::NETWORK:
            out << "Network";
            break;
        case LayerType::XNOR:
            out << "Xnor";
            break;
        case LayerType::REGION:
            out << "Region";
            break;
        case LayerType::YOLO:
            out << "Yolo";
            break;
        case LayerType::GAUSSIAN_YOLO:
            out << "Gaussian_yolo";
            break;
        case LayerType::ISEG:
            out << "Iseg";
            break;
        case LayerType::REORG:
            out << "Reorg";
            break;
        case LayerType::REORG_OLD:
            out << "Reorg_old";
            break;
        case LayerType::UPSAMPLE:
            out << "Upsample";
            break;
        case LayerType::LOGXENT:
            out << "Logxent";
            break;
        case LayerType::L2NORM:
            out << "L2norm";
            break;
        case LayerType::EMPTY:
            out << "Empty";
            break;
        case LayerType::BLANK:
            out << "Blank";
            break;
        case LayerType::CONTRASTIVE:
            out << "Contrastive";
            break;
        case LayerType::INPUT:
            out << "Input";
            break;
        }
        return out;
    }


    PaddingType paddingFromString(std::string pad)
    {
        if (pad == "valid")
            return PaddingType::VALID;
        else if (pad == "same")
            return PaddingType::SAME;
        throw std::runtime_error("Invalid padding type: '" + pad + "'");
    }
    std::ostream& operator<< (std::ostream& out, const PaddingType& obj)
    {
        if (obj == PaddingType::VALID)
            out << "valid";
        else if (obj == PaddingType::SAME)
            out << "same";
        return out;
    }

    template<> DataType getType<half>() { return DataType::DT_FLOAT16; }
    template<> DataType getType<float>() { return DataType::DT_FLOAT32; }
    template<> DataType getType<double>() { return DataType::DT_FLOAT64; }
    template<> DataType getType<int8_t>() { return DataType::DT_INT8; }
    template<> DataType getType<int32_t>() { return DataType::DT_INT32; }
    template<> DataType getType<int64_t>() { return DataType::DT_INT64; }
    template<> DataType getType<uint8_t>() { return DataType::DT_UINT8; }
    template<> DataType getType<uint32_t>() { return DataType::DT_UINT32; }
    template<> DataType getType<uint64_t>() { return DataType::DT_UINT64; }

} // namespace darknet
