#pragma once


namespace darknet
{
namespace params
{

    class layerParams
    {
    private:

    public:

        int stopbackward;
        layerParams() : stopbackward(0) {}
        ~layerParams() {}
    };

} // namespace params
} // namespace darknet