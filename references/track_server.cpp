#include "serial_cpp/serial.h" // https://www.github.com/wjwwood/serial
#include "mak_packet.h"
#include "cxxopts/cxxopts.hpp" // https://www.github.com/jarro2783/cxxopts
#include <CombinedApi.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sys/ioctl.h>
#include <string>
#include <cstring>
#include <thread>
#include <mutex>
#include <cassert>
#include <signal.h>
#include <vector>

#define AURORA_PORT_DESC "NDI Aurora"
#define CLIENT_PORT_DESC "Prolific Technology"
#define REQUIRE_AURORA true
#define REQUIRE_CLIENT false
#define USE_STATIC_PORTS true
#define STATIC_PORT_CLIENT ""   // make this empty string if not using client!
#define STATIC_PORT_AURORA "/dev/ttyUSB0"
#define BAUDRATE 115200U
#define BYTE_BUFFER_LENGTH 2048
#define PACKET_BUFFER_LENGTH 255
#define assertm(exp, msg) assert(((void)msg, exp))
#define MAX_PROBES 4

using namespace std;
using namespace serial_cpp;


// TODO: refactor into headers, etc.
// simple class for representing a serial message
class SimpleMsg {
    private:
        uint8_t msg_type;
        size_t msg_size;
        uint8_t *msg;
    public:

        // constructor
        SimpleMsg(uint8_t mtype, size_t msize, uint8_t* mdata){
            msg_type = mtype;
            msg_size = msize;
            msg = (uint8_t*) malloc(msize);
            memcpy(msg, mdata, msize);

        }

        // copy constructor 
        SimpleMsg(const SimpleMsg& orig){
            msg_type = orig.msg_type;
            msg_size = orig.msg_size;
            msg = (uint8_t*) malloc(msg_size);
            memcpy(msg, orig.msg, msg_size);
        }

        // destructor - frees array memory upon destruction of the SimpleMsg object
        ~SimpleMsg(){
            free(msg);
        }
        uint8_t get_msg_type(){
            return this->msg_type;
        }
        uint8_t* get_msg_buffer(){
            return this->msg;
        }
        size_t get_msg_size(){
            return this->msg_size;
        }
};

// globals
// TODO: rewrite to avoid using globals
std::vector<SimpleMsg> msg_buffer;
std::mutex msg_lock;
static CombinedApi capi = CombinedApi();
//static bool apiSupportsBX2 = false;
//static bool apiSupportsStreaming = false;
volatile bool exitflag = false;

// function prototypes
void monitor_serial_port(Serial* mySerialPort);
void onErrorPrintDebugMessage(std::string methodName, int errorCode);
std::string getToolInfo(std::string toolHandle);
void initializeAndEnableTools(std::vector<ToolData>& enabledTools);
void sig_callback_handler(int signum);

// main
int main(int argc, char** argv){


    uint8_t mypacket[MAX_PACKET_LENGTH] = {0};
    size_t mypacket_length = MAX_PACKET_LENGTH;
    uint32_t frame_num, probe_sn;
    uint32_t probes_requested[MAX_PROBES] = {0};
    uint8_t num_probes_requested = 0;
    std::vector<ToolData> enabledTools = std::vector<ToolData>();

    //uint8_t tool_num;
    //float q[SIZE_Q] = {0.0F};
    //float t[SIZE_T] = {0.0F};
    int result;
    bool capture_requested = false;
    uint8_t data_size;
    uint8_t *packet_buffer;
    char * p_phcstr;  // pointer to port handle c string
    char * p_reqcstr; // pointer to requested probe c string

    std::thread *serial_thread;

    // parse command line options
    // the add_options() syntax is a little strange, see: https://stackoverflow.com/questions/50844804
    bool wait_for_keypress = false;
    bool listen_mode = false;

    cxxopts::Options options("track_server","Simple serial interface to NDI Aurora");
    options.add_options()
        ("m,manual", "Do not stream, instead prompt to press enter between captures")
        ("l,listen", "Do not stream, instead wait for a serial request")
        ("o,output", "Filename for output",cxxopts::value<std::string>());
    options.parse_positional({"output"});
    auto cxxopts_result = options.parse(argc,argv);

    if(cxxopts_result.count("manual")){
        cout << "Manual acquisition mode!" << endl;
        wait_for_keypress = true;
    } else if(cxxopts_result.count("listen")){
        cout << "Listen acquisition mode!" << endl;
        listen_mode = true;
    }

    // initialize output file
    ofstream outfile;
    if(cxxopts_result.count("output")){
        cout << "Writing output to: " << cxxopts_result["output"].as<std::string>() << endl;
        outfile.open(cxxopts_result["output"].as<std::string>());
    } else {
        cout << "Writing output to: output.csv" << endl;
        outfile.open("output.csv");
        outfile << "aurora_time,aurora_time2,tool_id,q0,q1,q2,q3,tx,ty,tz,tx_tip,ty_tip,tz_tip,reg_error,capture_note" << endl;
    }

    // ensure that we're on a linux system
    // TODO: handle Windows, darwin, etc.
    if(!__linux__){
        printf("Error: not running on a linux system\n");
        return -1;
    }

    // ensure width of float type is four bytes
    // TODO: handle other cases
    if(sizeof(float) != 4){
        printf("Error: float datatype is not four bytes wide\n");
        return -1;
    }
    //static_assert(sizeof(float)==4,"Float size is not four bytes!"); 

    // ensure little endian encoding
    // TODO: handle big endian case
    static float test_float = 1.254;
    uint8_t* p_fval = (uint8_t*) &test_float;
    if(*p_fval != 0x12){
        printf("Error: not a litle endian system\n");
        return -1;
    }

    // open aurora and client serial ports, and list them if requested 
    std::string aurora_port_string = "";
    std::string client_port_string = "";
    if(USE_STATIC_PORTS){
        aurora_port_string = std::string(STATIC_PORT_AURORA);
        client_port_string = std::string(STATIC_PORT_CLIENT);
    } else {
        vector<PortInfo> all_ports = list_ports();
        if (all_ports.size() > 0) {

            // look through all COM ports
            cout << "Searching available COM ports..." << endl;
            for (unsigned int i = 0; i < all_ports.size(); i++) {

                // check for Aurora port
                if( all_ports[i].description.find(AURORA_PORT_DESC) != std::string::npos){
                    if(!aurora_port_string.empty()){
                        printf("Error: multiple Aurora ports found!\n");
                        return -1;
                    }
                    aurora_port_string = all_ports[i].port;
                    cout << "Found Aurora on " << aurora_port_string << endl;
                }

                // check for client port
                if( all_ports[i].description.find(CLIENT_PORT_DESC) != std::string::npos){
                    if(!client_port_string.empty()){
                        printf("Error: multiple client ports found!\n");
                        return -1;
                    }
                    client_port_string = all_ports[i].port;
                    cout << "Found client on " << client_port_string << endl;
                }

            }

            // make sure we've found both a client and an Aurora port
            if( aurora_port_string.empty() ){
                cout << "Error: no Aurora port found" << endl;
                if(REQUIRE_AURORA)
                    return -1;
            }
            if( client_port_string.empty() ) {
                cout << "Error: no client port found" << endl;
                if(REQUIRE_CLIENT) 
                    return -1;
            }


        } else {
            cout << "No ports found!" << endl;
        }
    }

    // open client serial port (not the tracker)
    Serial* mySerialPort = NULL;
    if(!client_port_string.empty()){
        cout << "Attempting to open connection to client on " << client_port_string << endl;
        try {
            mySerialPort = new Serial(client_port_string, BAUDRATE, Timeout(Timeout::max(),4,0,4,0), eightbits, parity_none, stopbits_one, flowcontrol_none);
        } catch(IOException const& e){
            cout << e.what() << endl;
            return -1;
        };
        mySerialPort->flush();
        cout << "done." << endl;
       
       // start thread to read messages from the client
       printf("Starting thread to read from client serial connection...\n");
       serial_thread = new std::thread(monitor_serial_port,mySerialPort);
    }


    // open aurora serial port
    if(capi.connect(aurora_port_string) != 0){
    printf("Error connecting to NDI API...\r");
    }

    // sleep for a second (NDI required?)
    sleep(1);    

    // get NDI device firmware version?
    cout << capi.getUserParameter("Features.Firmware.Version") << endl;

    // determine support for BX2

    // attempt to initialize NDI common API
    if( (result = capi.initialize()) < 0 ){
        cout << "Error initializing NDI common api: " << capi.errorToString(result) << endl;
        return -1;
    }

    // THIS SLEEP IS SUPER IMPORTANT!
    // WITHOUT IT PORT HANDLES ARE NOT DISCOVERED CORRECTLY
    sleep(1);

    cout << capi.getTrackingDataTX() << endl;


    // initialize and enable tools
    initializeAndEnableTools(enabledTools);
    cout << "Number of tools found: " << enabledTools.size() << endl;
    for(unsigned long tid = 0; tid < enabledTools.size(); ++tid){
        cout << tid << " -> " << enabledTools[tid].toolInfo << endl;
        //cout << tid << " -> " << enabledTools[tid].getSerialNumber() << endl;
    }

    // enter tracking mode
    cout << "Entering tracking mode" << endl;
    if( capi.startTracking() < 0 ){
    cout << "Error entering tracking mode." << endl;
    return -1;
    }

    // register signal handler to catch CTRL-C
    signal(SIGINT, sig_callback_handler);


    if(wait_for_keypress){
        cout << "Press ENTER to capture a transform, CTRL+C to end...";
    } else if(listen_mode){
        cout << "Listening for transform requests via serial link, CTRL+C then ENTER to end..." << endl;
    } else {
        cout << "Streaming transforms via serial link, CTRL+C to end..." << endl;
    }

    // start data capture loop
    uint32_t prev_frame_num = 0;
    long unsigned int cap_num = 0;
    while(!exitflag){ 
        ++cap_num;    

        if(wait_for_keypress){
            std::cin.get();
        }

        if(listen_mode){
            capture_requested = false;
            num_probes_requested = 0;
            while(!capture_requested && !exitflag){
                if(msg_buffer.size() > 0){
                    msg_lock.lock();
                    assertm( msg_buffer.size() > 0, "Vector length changed while acquirng lock.." );
                    SimpleMsg this_msg(msg_buffer[0]);
                    msg_buffer.erase(msg_buffer.begin());
                    msg_lock.unlock();
                    packet_buffer = this_msg.get_msg_buffer();
                    capture_requested = true;

                    if(this_msg.get_msg_type() == PKT_TYPE_GET_PROBE_TFORM){

                        // TODO: SEND NAK IF ANY OF THESE ERRORS OCCUR

                        // get data payload size (number of transform serial numbers being transmitted)
                        if(capture_requested && (this_msg.get_msg_size() >= 5)){
                            memcpy(&data_size,packet_buffer+3,1);
                        } else {
                            printf("Error: incorrect packet length, discarding packet\n");
                            capture_requested = false;
                        }

                        // make sure data payload size is correct
                        if( capture_requested && (data_size > 4) ){
                            printf("Error: incorrect payload data size (%d), discarding packet\n",data_size);
                            capture_requested = false;
                        }

                        // make sure message length is correct
                        if( capture_requested && (this_msg.get_msg_size() != (long unsigned int)(7+4*data_size))){
                            printf("Error: wrong message size (%ld) for number of transforms requested, discarding packet\n",this_msg.get_msg_size());
                            capture_requested = false;
                        }

                        // get serial number of each requested probe and store in array
                        if(capture_requested){
                            printf("Requested transform for probes(s):");
                            for(uint8_t id_idx = 0; id_idx < data_size; ++id_idx){
                                memcpy((probes_requested+num_probes_requested),packet_buffer+4+4*id_idx,4);
                                p_reqcstr = (char*)(probes_requested+num_probes_requested);
                                printf(" %s",p_reqcstr);
                                //printf(" 0x%08X",*(probes_requested + num_probes_requested));
                                ++num_probes_requested;
                            }
                            printf("\n");
                        }

                    } else {
                        printf("Received a non-transform requet packet, discarding\n");
                    }
                }
            }
        }

        if( exitflag )
            break;

        // get a set of transforms
        std::vector<ToolData> newToolData = capi.getTrackingDataBX(TrackingReplyOption::TransformData | TrackingReplyOption::AllTransforms);
        //cout << "Size of new tool data vector: " << newToolData.size() << endl;

        // copy tool transform data into enabledTools vector
        // double loop structure from NDI sample program
        // likely b/c tool data may come in in a different order than the enabledTools vector?
        for(long unsigned int tool_idx = 0; tool_idx < enabledTools.size(); tool_idx++){
            for(long unsigned int data_idx = 0; data_idx < newToolData.size(); data_idx++){
                if(enabledTools[tool_idx].transform.toolHandle == newToolData[data_idx].transform.toolHandle){
                    newToolData[data_idx].toolInfo = enabledTools[tool_idx].toolInfo; // preserves serial number
                    enabledTools[tool_idx] = newToolData[data_idx];

                    //cout << "tool info: " << newToolData[data_idx].toolInfo << endl; 

                }
            }
        } 

        // construct and send a packet for each tool
        std::vector<tform> tforms;
        for(long unsigned int tool_idx = 0; tool_idx < enabledTools.size(); tool_idx++){

            // create a transform struct
            tform thistf;

            // get the current frame number
            frame_num = (uint32_t) enabledTools[tool_idx].frameNumber;

            // get tool serial number REALLY PORT NUMBER, convert from cstr to integer
            strcpy((char *)&probe_sn,enabledTools[tool_idx].toolInfo.c_str());
            p_phcstr = (char *)&probe_sn; 
            //probe_sn = (uint32_t)strtol(enabledTools[tool_idx].toolInfo.c_str()+3,nullptr,16);
            if(tool_idx == 0){           
                cout << "Caputure " << cap_num << ", Frame " << frame_num << " Tools in frame:" << endl;
            }
            //printf("%ld",tool_idx);
            printf("* %s",p_phcstr); // TODO: do this the c++ way with cout
            if( enabledTools[tool_idx].transform.isMissing() ){
                cout << " --> MISSING!" << endl;
            } else if( frame_num == prev_frame_num ){
                cout << " --> REPEAT!" << endl;
            } else {
                if(!wait_for_keypress || (tool_idx < enabledTools.size()-1)){
                    cout << endl;
                }

                // determine whether this was one of the transforms we requested
                bool this_packet_requested = false;
                for(uint8_t id_idx = 0; id_idx < num_probes_requested; ++id_idx){
                    p_reqcstr = (char*)(probes_requested+id_idx);                
                    if(!strcmp(p_phcstr,p_reqcstr)){
                    //if( *(probes_requested + id_idx) == probe_sn){
                        this_packet_requested = true;
                    }
                }

                // capture transform for transmission *unless* we're in listen mode and this wasn't in the request list
                if( !(listen_mode && !this_packet_requested) ){    
                    thistf.id = probe_sn;
                    thistf.q0 = (float)enabledTools[tool_idx].transform.q0;
                    thistf.q1 = (float)enabledTools[tool_idx].transform.qx;
                    thistf.q2 = (float)enabledTools[tool_idx].transform.qy;
                    thistf.q3 = (float)enabledTools[tool_idx].transform.qz;
                    thistf.t0 = (float)enabledTools[tool_idx].transform.tx;
                    thistf.t1 = (float)enabledTools[tool_idx].transform.ty;
                    thistf.t2 = (float)enabledTools[tool_idx].transform.tz;
                    thistf.error = (float)enabledTools[tool_idx].transform.error;
                    tforms.push_back(thistf);
                    outfile << frame_num << "," << frame_num << "," << p_phcstr/*[strlen(p_phcstr)-1]*/ << ",";
                    outfile << thistf.q0 << "," << thistf.q1 << "," << thistf.q2 << "," << thistf.q3 << ",";
                    outfile << thistf.t0 << "," << thistf.t1 << "," << thistf.t2 << ",";
                    outfile << thistf.t0 << "," << thistf.t1 << "," << thistf.t2 << ",";
                    outfile << thistf.error << "," << endl;   
                } 
            }
        }

        // now send a response packet with however many transforms (0 or more) we have accumulated
        if(!client_port_string.empty()){            
        
        mypacket_length = MAX_PACKET_LENGTH;
        result = compose_tracker_packet(mypacket, &mypacket_length, frame_num, tforms);
        if( result != 0 ){
            printf("Error: unexpected result from packet composition (%d)\n",result);
            return -1;
        }
        //cout << "Writing packet to serial port..." << endl;
        mySerialPort->write(mypacket,mypacket_length);
        }
    }

    cout << "Stopping tracking" << endl;
    //onErrorPrintDebugMessage("capi.stopTracking()", capi.stopTracking());
    if( capi.stopTracking() < 0 ){
        cout << "Error stopping tracking" << endl;
        return -1;
    }

    // close output file
    outfile.close();


    // close serial port
    if(!client_port_string.empty()){            
        // join read thread
        serial_thread->join();
        cout << "Closing serial port" << endl;
        mySerialPort->close();
    }

    // done	
    return 0;
}

// function from NDI to show error codes on failure
void onErrorPrintDebugMessage(std::string methodName, int errorCode)
{
    if (errorCode < 0)
    {
        std::cout << methodName << " failed: " << capi.errorToString(errorCode) << std::endl;
    }
}

// function from NDI to get tool info
std::string getToolInfo(std::string toolHandle)
{
    // Get the port handle info from PHINF
    PortHandleInfo info = capi.portHandleInfo(toolHandle);

    // Return the ID and SerialNumber the desired string format
    std::string outputString = info.getToolId();
    //outputString.append(" s/n:").append(info.getSerialNumber());
    outputString.append(info.getSerialNumber());
    return outputString;
}

// function from NDI to initialize and enable tools
void initializeAndEnableTools(std::vector<ToolData>& enabledTools)
{
    std::cout << std::endl << "Initializing and enabling tools..." << std::endl;

    // Initialize and enable tools
    std::vector<PortHandleInfo> portHandles = capi.portHandleSearchRequest(PortHandleSearchRequestOption::NotInit);
    cout << "Number of port handles found: " << portHandles.size() << endl;
    for (long unsigned int i = 0; i < portHandles.size(); i++)
    {
        onErrorPrintDebugMessage("capi.portHandleInitialize()", capi.portHandleInitialize(portHandles[i].getPortHandle()));
        onErrorPrintDebugMessage("capi.portHandleEnable()", capi.portHandleEnable(portHandles[i].getPortHandle()));
    }

    // Print all enabled tools
    portHandles = capi.portHandleSearchRequest(PortHandleSearchRequestOption::Enabled);
    for (long unsigned int i = 0; i < portHandles.size(); i++)
    {
        std::cout << portHandles[i].toString() << std::endl;
    }

    // Lookup and store the serial number for each enabled tool
    for (long unsigned int i = 0; i < portHandles.size(); i++)
    {
        enabledTools.push_back(ToolData());
        enabledTools.back().transform.toolHandle = (uint16_t)capi.stringToInt(portHandles[i].getPortHandle());
        
        // 15-SEP-23: NDI (mostly) confirmed that our tools won't have S/Ns, so we're loading the two-byte port index string into toolInfo
        enabledTools.back().toolInfo = portHandles[i].getPortHandle();  //getToolInfo(portHandles[i].getPortHandle());
        //cout << "FOUND TOOL S/N: " << portHandles[i].getSerialNumber() << endl;
        //cout << "string length: " << portHandles[i].getSerialNumber().length() << endl;
        //cout << "getPortHandle() = " << portHandles[i].getPortHandle() << endl;
    }
}

// function to read from serial port, to be run in dedicated thread
void monitor_serial_port(Serial* mySerialPort){

    uint8_t byte_buffer[BYTE_BUFFER_LENGTH] = {0};
    size_t bytes_read;
    uint8_t *p_byte_end = byte_buffer + BYTE_BUFFER_LENGTH;
    uint8_t *p_byte, *p_start_search, *p_stop_search, *p_byte_copy;
    uint8_t start_found_flag, end_found_flag, crc_verified_flag;
    uint8_t DLE_count = 0;
    uint32_t packet_length;
    uint8_t packet_buffer[PACKET_BUFFER_LENGTH] = {0};
    uint8_t *p_packet = packet_buffer;
    uint8_t *p_packet_end = packet_buffer + PACKET_BUFFER_LENGTH;
    uint8_t CRC_received, CRC_computed;
    uint8_t packet_type;
    uint8_t *p_display;

    // start adding to beginning of byte buffer
    p_byte = byte_buffer;

    // loop forever, capturing bytes from serial stream and
    // processing packets when they are identified
    printf("Parsing data from serial port...\n");
    while(!exitflag){

        // reset all parsing variables
        p_start_search = byte_buffer;
        DLE_count = 0;
        start_found_flag = 0;
        end_found_flag = 0;
        packet_length = 0;
        crc_verified_flag = 0;

        // read all available bytes from serial port
        // TODO: this will need to be updated for implementation on a microcontroller
        bytes_read = mySerialPort->read(p_byte,p_byte_end-p_byte);
        p_byte += bytes_read;

        // try to position p_start_search at DLE byte in DLE STX 
        while( !start_found_flag && p_start_search < p_byte) {
            while(*p_start_search == DLE){
                ++DLE_count;
                ++p_start_search;
            }
            if( (DLE_count % 2 == 1) && *p_start_search == STX ){
                start_found_flag = 1;
            } else {
                ++p_start_search;
                DLE_count = 0;
            }
        }

        // adjust p_byte or p_start_search
        if( !start_found_flag ){
            //printf("Start not found!\n");
            // we didn't find a start, so clear byte buffer
            // unless last byte is DLE which could be part of a start byte
            if(p_byte > byte_buffer){
                if( *(p_byte-1) == DLE ){
                    *(byte_buffer) = DLE;
                    p_byte = byte_buffer + 1;
                } else {
                    p_byte = byte_buffer;
                }
            }
            printf("p_byte is %ld bytes ahead of buffer start\n",p_byte-byte_buffer);
        } else {
            //printf("Found start\n");
            //back up pointer to the DLE
            --p_start_search;

            /* DEBUG */ 
               printf("Byte buffer from start of message: ");
               p_display = p_start_search;
               while(p_display < p_byte){
               printf("0x%02X ",*p_display);
               ++p_display;
               }
               printf("\n");
               /**/
        }

        // try to find DLE ETX
        // in a new block to reduce nesting
        if(start_found_flag){
            p_stop_search = p_start_search;
            DLE_count = 0;

            while( !end_found_flag && (p_stop_search < p_byte)) {
                while(*p_stop_search == DLE){
                    ++DLE_count;
                    ++p_stop_search;
                }
                if( (DLE_count % 2 == 1) && (*p_stop_search == ETX) ){
                    end_found_flag = 1;
                } else {
                    ++p_stop_search;
                    DLE_count = 0;
                }
            }

            /* DEBUG
               if(end_found_flag){
               printf("Found end\n");
               } else {
               printf("No end found\n");
               }
               */
        }

        // now we have a full message
        // so copy it out to message buffer and reset the byte buffer
        // while copying to byte buffer un-stuff DLEs
        if(end_found_flag){

            // increment p_stop_search so it points to one byte past ETX
            ++p_stop_search;

            // copy each byte out
            p_packet = packet_buffer;
            p_byte_copy = p_start_search;
            while(p_byte_copy < p_stop_search){

                // make sure we have enough room to copy another byte into the packet buffer
                if(p_packet >= p_packet_end){
                    printf("Error - insufficient message buffer length\n");
                    return;
                }

                // copy this byte in
                *p_packet = *p_byte_copy;
                ++packet_length;

                // increment packet buffer pointer
                ++p_packet;

                // increment byte buffer pointer, un-stuffing DLEs
                if((*p_byte_copy == DLE) && ((p_byte_copy+1) < p_byte_end) && (*(p_byte_copy+1) == DLE) ){
                    p_byte_copy = p_byte_copy + 2;
                } else {
                    ++p_byte_copy;
                }
            }

            /* DEBUG
            // now we have a complete message
            // display it
            p_display = packet_buffer;
            printf("Unstuffed packet: ");
            while(p_display < (packet_buffer + packet_length)){
            printf("0x%02X ",*p_display);
            ++p_display;
            }
            printf("\n");
            */

            // reset byte buffer
            // i.e. copy everything past message we extracted back to the beginning of the buffer
            p_byte_copy = byte_buffer;
            while(p_stop_search < p_byte){
                *p_byte_copy = *p_stop_search;
                ++p_byte_copy;
                ++p_stop_search;
            }
            p_byte = p_byte_copy;

            /* DEBUG
            // now show remaining buffer past the packet we just extracted
            p_display = byte_buffer;
            printf("Remaining bytes in byte_buffer: ");
            while(p_display < p_byte){
            printf("0x%02X ",*p_display);
            ++p_display;
            }
            printf("\n");
            */

        }

        // now we have an unstuffed packet
        // check CRC
        if(packet_length > 0){

            // ensure packet is at least the minimum length required
            if(packet_length < 6){
                printf("Error: packet not long enough to have a CRC and length byte"); // length = 4 would be {DLE,STX,DLE,ETX}
            return;
            }

            // extract CRC from packet
            CRC_received = *(packet_buffer + packet_length -3);

            // compute CRC
            CRC_computed = 0x00;
            crcAddBytes(&CRC_computed,packet_buffer+2,packet_length-5); // adjust to omit DLE, STX, ETX, and CRC8 bytes

            // check CRC
            //printf("CRC8 received: 0x%02X, computed: 0x%02X\n",CRC_received,CRC_computed);
            if(CRC_received != CRC_computed){
                printf("Error: CRC8 mismatch, discarding packet\n");
            } else {
                crc_verified_flag = 1;
            }
        }

        // process the packet we've received
        if(crc_verified_flag){

            // get packet type
            packet_type = *(packet_buffer + 2);

            //printf("Received packet of type 0x%02X and length %d\n",packet_type,packet_length);

            // create a new message object and put it into vector           
            SimpleMsg new_msg(packet_type,packet_length,packet_buffer);
            msg_lock.lock();
            msg_buffer.push_back(new_msg);
            msg_lock.unlock();


        }
    }
    return;
}

// signal handler to stop data collection, etc. on CTRL-C
void sig_callback_handler(int signum){
    exitflag = true;
    return;
}

